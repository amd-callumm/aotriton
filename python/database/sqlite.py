# Copyright © 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import sqlite3
import pandas as pd
from ..template_instantiation.ir import typed_choice as TC
from ..utils import log
from ..gpu_targets import AOTRITON_TUNING_DATABASE_REUSE

'''
We don't really need a LazyTableView, if Lazy evaluation is needed, a
LazyPandasDataFrame is more preferrable
'''
# from .view import LazyTableView as SqliteTableView

def create_select_stmt(table_name, wheres):
    stmt = f"SELECT * FROM {table_name} WHERE "
    where_stmt = []
    params = []
    for k, v in wheres.items():
        if isinstance(v, list) or isinstance(v, tuple):
            qm = ', '.join(['?'] * len(v))
            where_stmt.append(f'{k} IN ({qm})')
            params += v
        else:
            where_stmt.append(f'{k} = ?')
            params.append(v.sql_value if isinstance(v, TC.TypedChoice) else v)
    stmt += ' AND '.join(where_stmt)
    # print('create_select_stmt', stmt)
    return stmt, params

def format_sql(stmt, params):
    template = stmt.replace('?', '{!r}')
    return (stmt, params)

class Factory(object):
    SIGNATURE_FILE = 'database/tuning_database.sqlite3'
    SECONDARY_DATABASES = {
        'op': 'database/op_database.sqlite3',
    }

    def __init__(self, path):
        log(lambda : f'sqlite3.connect({path / self.SIGNATURE_FILE})')
        self._conn = sqlite3.connect(path / self.SIGNATURE_FILE)
        self._conn.set_trace_callback(log) # Debug
        for schema, bn in self.SECONDARY_DATABASES.items():
            fn = path / bn
            if fn.is_file():
                log(lambda : f"ATTACH DATABASE '{fn.as_posix()}' AS {schema};")
                self._conn.execute(f"ATTACH DATABASE '{fn.as_posix()}' AS {schema};")
            else:
                assert False, f'{fn} is not a file, {path}'

    def create_view(self, functional):
        """Query tuning database with N-tier GPU prioritization.

        For each target GPU, iterates its priority chain and returns the first
        match. Results are tagged with target_gpu for caller to filter per-GPU
        LUT slices. Falls back to fallback_choices if compact_choices yields nothing.
        """
        log(lambda : f'{functional=}')
        meta = functional.meta_object
        pfx = 'op.' if getattr(meta, 'CODEGEN_MODULE', None) == 'op' else ''
        table_name = pfx + meta.FAMILY.upper() + '$' + meta.NAME

        # Get GPU priority chains for each target GPU
        # e.g. (multi-mod): {'gfx1151_mod0': [...], 'gfx1151_mod1': [...]}
        # e.g. (multi-arch): {'gfx1151_mod0': [...], 'gfx1150_mod0': [...]}
        target_priority_map = functional.database_gpus

        def build_sql(choice_dict):
            """Query each target GPU's priority chain, return combined results.

            Returns (dataframe, sql) where dataframe has target_gpu column.
            """
            all_dfs = []
            last_sql = None

            for target_gpu, priority_chain in target_priority_map.items():
                # Try each GPU in this target's priority chain
                for db_gpu in priority_chain:
                    # Build WHERE clause for this GPU and configuration
                    wheres = {'gpu': db_gpu}
                    for key, value in choice_dict.items():
                        if isinstance(value, TC.TypedChoice) and value.is_tensor:
                            wheres[f'inputs${key}_dtype'] = value
                        else:
                            wheres[f'inputs${key}'] = value

                    stmt, params = create_select_stmt(table_name, wheres)
                    try:
                        log(lambda : f'Trying {db_gpu} for target {target_gpu}: {stmt}')
                        df = pd.read_sql_query(stmt, self._conn, params=params)
                        if not df.empty:
                            # Tag rows with target GPU for filtering
                            df['target_gpu'] = target_gpu
                            all_dfs.append(df)
                            last_sql = format_sql(stmt, params)
                            log(lambda : f'Found {len(df)} rows in {db_gpu} for {target_gpu}')
                            break  # Found match for this target, move to next target
                    except pd.errors.DatabaseError:
                        log(lambda : f'Table {table_name} may not exist')
                        return None, format_sql(stmt, params)

            if all_dfs:
                return pd.concat(all_dfs, ignore_index=True), last_sql
            return pd.DataFrame(), None

        # Try exact match first (compact_choices)
        df, sql = build_sql(functional.compact_choices)
        if df is None:
            # Database error
            return df, sql
        if not df.empty:
            return df, sql

        # No exact match - try fallback_choices
        return build_sql(functional.fallback_choices)
