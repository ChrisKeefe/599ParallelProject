# 599ParallelProject

Semester Project and documentation for CS599: Parallel Programming

## Sequential LLoyd's

- Navigate into `lloyds_seq`
- Run on iris data set with five cluster centers using `make run k=5` or...
- Compile with `make` and run as follows:
`./lloyds_seq k=<n_centers> <filepath_to_data> <data_delimiter> <contains_header> <drop_labels>

where:
`n_centers` is an integer number of cluster centers
`filepath_to_data` is an absolute or relative filepath
`data_delimiter` is the delimiter used to separate records in your `.csv`, `.tsv`, etc.
`contains_header` is 1 if your data contains a header row, 0 if not
`drop_labels` is 1 if the last column of your data contains labels, 0 if not (labels must be dropped - only numerical data is allowed)