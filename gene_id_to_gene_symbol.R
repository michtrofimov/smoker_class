library(dplyr)
library(data.table)
library(AnnotationHub)
library(ensembldb)
library(stringr)
library(DESeq2)
library(tibble)
library(EnhancedVolcano)
library(clusterProfiler)
library(org.Hs.eg.db)
library(sigPathway)

# set working directory
main_dir <- dirname(rstudioapi::getSourceEditorContext()$path) 
setwd(main_dir)

# create annotationhub dataframe
ah = AnnotationHub()

# choose human Ensemble database
human_db = query(ah, c('Homo sapiens', 'EnsDb'))

# extract latest human database
human_db = human_db[["AH113665"]]

# extract gene names
gene_db = genes(human_db, return.type = "data.frame")
gene_db_protein_coding = gene_db %>%
  filter(gene_biotype == "protein_coding")

# upload counts for all bams (GRCh38)
df = fread('expression/counts/counts.txt')

# choose only bams for QKI (hepg2) depletion (batch1)
qki = df[,c('Geneid','ENCFF069VOQ.bam', 'ENCFF827WGG.bam', 'ENCFF694JWV.bam','ENCFF902YHV.bam')]


# strip versions of gene_id
qki$Geneid <- str_replace(qki$Geneid,
                          pattern = ".[0-9]+$",
                          replacement = "")

# merge by gene_id
qki = inner_join(qki, gene_db_protein_coding, by = c("Geneid" = 'gene_id'))

# if no gene name, drop
qki <- qki %>%
  mutate(across(where(is.character), ~ na_if(.,"")))

qki = qki %>% filter_at(vars("symbol"), all_vars(!is.na(.)))

# choose expressions, gene_id and gene_name column
qki = qki[,2:6]

# how many duplicated genes
table(duplicated(qki$gene_name))
sum(is.na(qki$gene_name))

# remove duplicates
qki = qki[!duplicated(qki$gene_name),]

# check na values
qki[!complete.cases(qki),]

# remove na entries
qki = qki[complete.cases(qki),]

# set index with gene name
qki <- qki %>% column_to_rownames(., var = 'gene_name')
