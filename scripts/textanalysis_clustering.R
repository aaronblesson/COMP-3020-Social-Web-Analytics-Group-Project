# Text analysis and clustering

# load the data
posts = read.csv(
  "finalhandover_corrected.csv",
  fileEncoding = "UTF-8",
  stringsAsFactors = FALSE
)

dim(posts)
names(posts)

# checking text and labels
sum(is.na(posts$text))
sum(posts$text == "", na.rm = TRUE)

sum(duplicated(posts$uri))

sum(duplicated(posts$text))

table(posts$sentiment)

# load text packages
library(tm)
library(SnowballC)

# keep the original text
posts_text = posts$text

Bluesky.corpus = Corpus(VectorSource(posts_text))

nrow(posts)
length(Bluesky.corpus)
content(Bluesky.corpus[[1]])

# remove links
remove_urls = content_transformer(function(x) {
  gsub(
    paste0(
      "https?://\\S+|www\\.\\S+|\\b[A-Za-z0-9-]+(?:\\.[A-Za-z0-9-]+)*\\.",
      "[A-Za-z]{2,}/\\S*"
    ),
    " ",
    x,
    perl = TRUE
  )
})

corpus = tm_map(Bluesky.corpus, remove_urls)

length(corpus)

# checking
link_positions = grep(
  paste0(
    "https?://\\S+|www\\.\\S+|\\b[A-Za-z0-9-]+(?:\\.[A-Za-z0-9-]+)*\\.",
    "[A-Za-z]{2,}/\\S*"
  ),
  posts$text,
  perl = TRUE
)

for (i in head(link_positions, 2)) {
  print(paste("Post row:", i))
  
  print("Original:")
  print(posts$text[i])
  
  print("After link removal:")
  print(content(corpus[[i]]))
}

# simplify characters
corpus = tm_map(
  corpus,
  function(x) iconv(x, to = "ASCII", sub = " ")
)

# clean and stem the text
corpus = tm_map(corpus, removeNumbers) # removes digits
corpus = tm_map(corpus, removePunctuation) # Remove punctuation
corpus = tm_map(corpus, stripWhitespace) # Trim spaces
corpus = tm_map(corpus, tolower) # Lowercase
corpus = tm_map(corpus, removeWords, stopwords("english")) # Remove common words
corpus = tm_map(corpus, stemDocument) # Stem words

posts$text[2]
content(corpus[[2]])

length(corpus)

# word-count matrix
Bluesky.dtm = DocumentTermMatrix(corpus)
Bluesky.matrix = as.matrix(Bluesky.dtm)

dim(Bluesky.matrix)

Bluesky.matrix[1:3, 1:6]

nd = rowSums(Bluesky.matrix)

sum(nd == 0)

# inspecting empty documents
posts_zero_terms = posts[nd == 0, c("uri", "text")]

posts_zero_terms

# keep posts with usable words
keep = nd > 0

posts_text_analysis = posts[keep, ]
Bluesky.matrix = Bluesky.matrix[keep, , drop = FALSE]

nd = rowSums(Bluesky.matrix)

dim(posts_text_analysis)
dim(Bluesky.matrix)
sum(nd == 0)

# count frequent words
w = colSums(Bluesky.matrix)

o = order(w, decreasing = TRUE)[1:20]

w[o]

#pPlot frequent words
library(wordcloud)

set.seed(3020)

wordcloud(
  words = names(w),
  freq = w,
  random.order = FALSE,
  min.freq = 6,
  max.words = 100,
  colors = brewer.pal(8, "Dark2"),
  scale = c(3, 0.7)
)

title("Frequent terms in political and social discussion")

# calculating TF-IDF weights
N = nrow(Bluesky.matrix)

IDF = log(N / colSums(Bluesky.matrix > 0))

TF = log(Bluesky.matrix + 1)

Bluesky.weighted.matrix = t(t(TF) * IDF)

dim(Bluesky.weighted.matrix)
Bluesky.weighted.matrix[1:3, 1:6]

# compare word rankings
w_tfidf = colSums(Bluesky.weighted.matrix)

o_tfidf = order(w_tfidf, decreasing = TRUE)[1:20]

w_tfidf[o_tfidf]

w[o]

# plot weighted words
set.seed(3020)

wordcloud(
  words = names(w_tfidf),
  freq = w_tfidf,
  random.order = FALSE,
  min.freq = 6,
  max.words = 100,
  colors = brewer.pal(8, "Dark2"),
  scale = c(2.8, 0.5)
)

title("TF-IDF weighted terms in political and social discussion")

# principal component analysis (PCA)
pcaT = prcomp(Bluesky.weighted.matrix)

summary(pcaT)$importance[, 1:5]

# plot first two components
plot(
  pcaT$x[, 1], pcaT$x[, 2],
  col = "blue", pch = 16, cex = 0.5,
  xlab = "PC1 (1.495%)",
  ylab = "PC2 (1.376%)",
  main = "PCA of TF-IDF weighted posts"
)

# inspect unusual post
outlier_position = order(pcaT$x[, 2])[1]

posts_text_analysis$text[outlier_position]

sum(posts_text_analysis$text ==
      posts_text_analysis$text[outlier_position],
    na.rm = TRUE
)

# cosine distances
tfidf_lengths = sqrt(rowSums(Bluesky.weighted.matrix^2))
sum(tfidf_lengths == 0)

stopifnot(all(tfidf_lengths > 0))

Bluesky.normalised.matrix =
  Bluesky.weighted.matrix / tfidf_lengths

summary(sqrt(rowSums(Bluesky.normalised.matrix^2)))

D_cosine = dist(Bluesky.normalised.matrix,
                method = "euclidean")^2 / 2

summary(D_cosine)

# mds dimensions
mds_check = cmdscale(D_cosine, k = 2, eig = TRUE)

summary(mds_check$eig)

positive_eigenvalues = mds_check$eig[mds_check$eig > 0]
positive_cumulative = cumsum(positive_eigenvalues) / sum(positive_eigenvalues)

dims_80 = which(positive_cumulative >= 0.80)[1]
dims_90 = which(positive_cumulative >= 0.90)[1]
c(dims_80 = dims_80, dims_90 = dims_90)

negative_share = 
  sum(abs(mds_check$eig[mds_check$eig < 0])) /
  sum(abs(mds_check$eig))
negative_share

# comparing cluster counts
mds_clustering = cmdscale(D_cosine, k = dims_80)
dim(mds_clustering)

set.seed(3020)
SSW_mds = rep(NA_real_, 15)

for (a in 1:15) {
  K_trial = kmeans(mds_clustering, centers = a, nstart = 20, iter.max = 100)
  SSW_mds[a] = K_trial$tot.withinss
}

plot(
  1:15, SSW_mds,
  type = "b",
  xlab = "Number of clusters",
  ylab = "Within-cluster sum of squares",
  main = "Elbow plot: cosine-distance MDS"
)

# inspect two clusters
set.seed(3020)
K2 = kmeans(
  mds_clustering,
  centers = 2,
  nstart = 20,
  iter.max = 100
)

table(K2$cluster)

for (cluster_number in 1:2) {
  cluster_ids = which(K2$cluster == cluster_number)
  cluster_weights = colMeans(
    Bluesky.weighted.matrix[cluster_ids, , drop = FALSE]
  )
  
  print(paste("Cluster", cluster_number))
  print(sort(cluster_weights, decreasing = TRUE)[1:10])
}

# petition texts
petition_posts = posts_text_analysis[K2$cluster == 2, ]

length(unique(petition_posts$text))

head(petition_posts$text, 3)

# four clusters
set.seed(3020)
K4 = kmeans(
  mds_clustering,
  centers = 4,
  nstart = 20,
  iter.max = 100
)

table(K4$cluster)

for (cluster_number in 1:4) {
  cluster_ids = which(K4$cluster == cluster_number)
  cluster_weights = colMeans(
    Bluesky.weighted.matrix[cluster_ids, , drop = FALSE]
  )
  
  print(paste("Cluster", cluster_number))
  print(sort(cluster_weights, decreasing = TRUE)[1:10])
}

# checking cluster
for (cluster_number in 2:4) {
  cluster_ids = which(K4$cluster == cluster_number)
  
  print(paste("Cluster", cluster_number))
  print(head(posts_text_analysis$text[cluster_ids], 3))
}

# six clusters
set.seed(3020)
K6 = kmeans(
  mds_clustering,
  centers = 6,
  nstart = 20,
  iter.max = 100
)

table(K6$cluster)

for (cluster_number in 1:6) {
  cluster_ids = which(K6$cluster == cluster_number)
  cluster_weights = colMeans(
    Bluesky.weighted.matrix[cluster_ids, , drop = FALSE]
  )
  
  print(paste("Cluster", cluster_number))
  print(sort(cluster_weights, decreasing = TRUE)[1:10])
}

# comparision of cluster and repeated text
for (cluster_number in c(1, 2, 3, 4)) {
  cluster_ids = which(K6$cluster == cluster_number)
  
  print(paste("Cluster", cluster_number))
  print(head(posts_text_analysis$text[cluster_ids], 2))
}

repeated_text = posts_text_analysis$text ==
  posts_text_analysis$text[outlier_position]

table(K4$cluster[repeated_text])
table(K6$cluster[repeated_text])

# use four-cluster solution
K_final = K4

cluster_titles = c(
  "Petition sharing",
  "Employment and labour markets",
  "Housing and accommodation",
  "Mixed political and economic discussion"
)

posts_text_analysis$cluster = K_final$cluster
posts_text_analysis$cluster_theme =
  cluster_titles[posts_text_analysis$cluster]

table(posts_text_analysis$cluster_theme)

# plot words in each cluster
set.seed(3020)

for (cluster_number in 1:4) {
  cluster_ids = which(K_final$cluster == cluster_number)
  
  cluster_weights = colMeans(
    Bluesky.weighted.matrix[cluster_ids, , drop = FALSE]
  )
  
  wordcloud(
    words = names(cluster_weights),
    freq = cluster_weights,
    min.freq = 0,
    max.words = 30,
    random.order = FALSE,
    colors = brewer.pal(8, "Dark2"),
    scale = c(3, 0.4)
  )
  
  title(paste("Cluster", cluster_number))
}

# plot final clusters
mds_plot = mds_check$points

plot(
  mds_plot[, 1], mds_plot[, 2],
  col = K_final$cluster,
  pch = 16,
  xlab = "MDS dimension 1",
  ylab = "MDS dimension 2",
  main = "Text clusters in political and social discussion",
  asp = 1
)

legend(
  "topright",
  legend = cluster_titles,
  col = 1:4,
  pch = 16,
  cex = 0.6
)

# sentiment labels count by cluster
table(
  Cluster = posts_text_analysis$cluster,
  Sentiment = posts_text_analysis$sentiment
)

# comparision of word presence by sentiment
negative_rows = posts_text_analysis$sentiment == "negative"
other_rows = posts_text_analysis$sentiment == "other"

negative_pct = 100 * colMeans(
  Bluesky.matrix[negative_rows, , drop = FALSE] > 0
)
other_pct = 100 * colMeans(
  Bluesky.matrix[other_rows, , drop = FALSE] > 0
)

head(sort(negative_pct, decreasing = TRUE), 10)
head(sort(other_pct, decreasing = TRUE), 10)

# compare leading terms
comparison_terms = unique(c(
  names(head(sort(negative_pct, decreasing = TRUE), 5)),
  names(head(sort(other_pct, decreasing = TRUE), 5))
))

round(cbind(
  Negative = negative_pct[comparison_terms],
  Other = other_pct[comparison_terms]
), 1)

# summarise reposts by cluster
for (cluster_number in 1:4) {
  for (sentiment_group in c("negative", "other")) {
    selected = posts_text_analysis$cluster == cluster_number &
      posts_text_analysis$sentiment == sentiment_group
    
    if (sum(selected) > 0) {
      reposts = posts_text_analysis$repost_count[selected]
      
      print(paste("Cluster", cluster_number, sentiment_group))
      print(round(c(
        Posts = length(reposts),
        Mean = mean(reposts),
        Median = median(reposts),
        Any_repost_pct = 100 * mean(reposts > 0)
      ), 2))
    }
  }
}

# collecting repost summaries
repost_summary = NULL
for (cluster_number in 1:4) {
  for (sentiment_group in c("negative", "other")) {
    selected = posts_text_analysis$cluster == cluster_number &
      posts_text_analysis$sentiment == sentiment_group
    if (sum(selected) > 0) { # skip groups with no posts
      reposts = posts_text_analysis$repost_count[selected]
      repost_summary = rbind(repost_summary, data.frame(
        Cluster = cluster_number, Sentiment = sentiment_group,
        N = length(reposts), Mean = round(mean(reposts), 2),
        Median = median(reposts),
        Any_pct = round(100 * mean(reposts > 0), 2)))
    }
  }
}
repost_summary 
