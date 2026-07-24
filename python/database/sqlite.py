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

        Implements two-stage selection strategy:
        1. Try compact_choices (exact shape) across GPU priority chain
        2. If no exact match found, try fallback_choices (similar shape)

        For each stage, tries GPUs in priority order defined by
        AOTRITON_TUNING_DATABASE_REUSE configuration.
        """
        log(lambda : f'{functional=}')
        meta = functional.meta_object
        pfx = 'op.' if getattr(meta, 'CODEGEN_MODULE', None) == 'op' else ''
        table_name = pfx + meta.FAMILY.upper() + '$' + meta.NAME

        # Get GPU priority chains for each target GPU
        # Currently single-mod per arch, so dict has one entry per Functional
        target_priority_map = functional.database_gpus
        # For single-mod case, just use the one priority chain
        _, priority_chain = next(iter(target_priority_map.items()))

        def build_sql(choice_dict):
            """Try each GPU in priority chain with given choice configuration.

            Returns (dataframe, sql) on first match, or (dataframe, None) if not found.
            """

            for gpu in priority_chain:
                # Build WHERE clause for this GPU and configuration
                wheres = {'gpu': gpu}
                for key, value in choice_dict.items():
                    if isinstance(value, TC.TypedChoice) and value.is_tensor:
                        wheres[f'inputs${key}_dtype'] = value
                    else:
                        wheres[f'inputs${key}'] = value

                # Query this GPU's database
                stmt, params = create_select_stmt(table_name, wheres)
                try:
                    log(lambda : f'Trying {gpu}: {stmt} params {params}')
                    df = pd.read_sql_query(stmt, self._conn, params=params)
                    if not df.empty:
                        log(lambda : f'Found tuning in {gpu}')
                        return df, format_sql(stmt, params)
                except pd.errors.DatabaseError:
                    log(lambda : f'Table {table_name} may not exist. stmt: {stmt} params {params}')
                    return None, format_sql(stmt, params)

            # No match found
            return pd.DataFrame(), None

        # Try exact match first (compact_choices)
        df, sql = build_sql(functional.compact_choices)
        if df is None:
            # Database error - return immediately
            return df, sql
        if not df.empty:
            # Found match with exact choices
            return df, sql

        # No exact match - try fallback_choices (e.g., PADDED_HEAD=False substitution)
        df, sql = build_sql(functional.fallback_choices)
        return df, sql
