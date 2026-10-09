library(igraph)

#gets Data
postData = read.csv("finalhandover100.csv", as.is = TRUE)
repostData = read.csv("cleanupreposttable.csv", as.is = TRUE)

# Nodes are accounts,Edges are from re-poster to post author
# post authors with no reposts aren't shown
el = cbind(from = repostData$reposterHandle, to = repostData$authorHandle)
el = unique(el)
el = el[el[, "from"] != el[, "to"], , drop = FALSE]
g = graph_from_edgelist(el, directed = TRUE)


# prints number of accounts, links, and density. 
vcount(g)
ecount(g)
round(edge_density(g), 4)

#shows in-degree
#mode = in to show reposters only
head(sort(degree(g, mode = "in"), decreasing = TRUE), 5)


# page rank, method adapted from labs
normalise = function(x) {
  if (sum(x) != 0) {
    return(x / sum(x))
  } else {
    return(rep(0, length(x)))
  }
}

adjacency.to.probability = function(A) {
  cols = ncol(A)
  for (a in 1:cols) {
    A[, a] = normalise(A[, a])
  }
  return(A)
}

difference = function(x, y) {
  return(sqrt(sum((x - y)^2)))
}

stationary.distribution = function(T) {
  n = ncol(T)
  p = rep(0, n)
  p[1] = 1
  p.old = rep(0, n)
  while (difference(p, p.old) > 1e-06) {
    p.old = p
    p = T %*% p.old
  }
  return(p)
}

pageRank = function(g) {
  A = t(as.matrix(as_adjacency_matrix(g, sparse = FALSE)))
  T = adjacency.to.probability(A)
  n = ncol(T)
  J = matrix(rep(1 / n, n * n), n, n)
  alpha = 0.8
  M = alpha * T + (1 - alpha) * J
  M = adjacency.to.probability(M)
  stationary.distribution(M)
}

# shows PageRank Results attached to account names
p = pageRank(g)
names(p) = V(g)$name
head(sort(p, decreasing = TRUE), 5)

#stores follower counts for each post author
followers = postData$followers_count[
  match(V(g)$name, postData$author_handle)]

#sets dot colors, orange for author, blue for reposter
V(g)$color = ifelse(
  V(g)$name %in% postData$author_handle, "coral3", "steelblue")

# Adjusts account node size based on follower count
# Missing follower nodes have a fixed size
V(g)$size = ifelse(
  is.na(followers), 2,
  2 + 8 * sqrt(followers / max(followers, na.rm = TRUE)))

#gets top account names, then label with corresponding rank for graph
topAccounts = names(head(sort(p, decreasing = TRUE), 5))
V(g)$label = ifelse(V(g)$name %in% topAccounts, match(V(g)$name, topAccounts), "")

#main repost network diagram 
set.seed(3020)
plot(g, layout = layout_with_fr(g, grid = "nogrid"),
     vertex.label.cex = 1.1,
     vertex.label.color = "black",
     vertex.label.font = 2,
     edge.arrow.size = 0.1,
     main = "Reposts Network")


#sub-graph for the largest connected portion of the main graph
g2 = largest_component(g, mode = "weak")
set.seed(3020)
plot(g2, layout = layout_with_kk(g2),
     vertex.label.cex = 1.1,
     vertex.label.color = "black",
     vertex.label.font = 2,
     edge.arrow.size = 0.1,
     main = "Largest Connected Group")