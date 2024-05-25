library(data.table)
gene_db = fread('gene_db.csv')
counts_bi = fread('../../../../BI/project/data/TCGA-LUAD.htseq_counts.tsv')
head(counts_bi)

gene_db_protein_coding = gene_db %>%
  filter(gene_biotype == "protein_coding")

# strip versions of gene_id
counts_bi$Ensembl_ID <- str_replace(counts_bi$Ensembl_ID,
                          pattern = ".[0-9]+$",
                          replacement = "")

# merge by gene_id
counts_bi_merged = inner_join(counts_bi, gene_db_protein_coding, by = c("Ensembl_ID" = 'gene_id'))
cols = c(2:586)
cols = c(cols, 597)
cols
counts_bi_merged_test = counts_bi_merged[,..cols]


# if no gene name, drop
counts_bi_merged_test <- counts_bi_merged_test %>%
  mutate(across(where(is.character), ~ na_if(.,"")))

counts_bi_merged_test = counts_bi_merged_test %>% filter_at(vars("symbol"), all_vars(!is.na(.)))

# how many duplicated genes
table(duplicated(counts_bi_merged_test$symbol))
sum(is.na(counts_bi_merged_test$symbol))

# remove duplicates
counts_bi_merged_test = counts_bi_merged_test[!duplicated(counts_bi_merged_test$symbol),]

fwrite(counts_bi_merged_test, '../../../../BI/project/data/TCGA-LUAD.htseq_counts.protein_coding.csv', )
