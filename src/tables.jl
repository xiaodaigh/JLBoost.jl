export nrow, ncol, view

import DataFrames: nrow, ncol, view, names

# convenience function
nrow(table) = length(Tables.rows(table))
ncol(table) = length(Tables.schema(table).names)

view(table, rows, cols) = map(Tables.columns(table)) do col
    @view col[rows]
end

names(table) = Tables.schema(table).names
