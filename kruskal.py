# Code in this file is copied from online
# Source : www.geeksforgeeks.org/kruskals-algorithm-simple-implementation-for-adjacency-matrix/

def find(i, parent):
    while parent[i] != i:
        i = parent[i]
    return i
 
# Does union of i and j. It returns
# false if i and j are already in same 
# set. 
def union(i, j, parent):
    a = find(i, parent)
    b = find(j, parent)
    parent[a] = b


# Finds MST using Kruskal's algorithm 
def kruskalMST(cost):
    edges = []

    V = 15
    parent = [i for i in range(V)]
    INF = float('inf')

    mincost = 0 # Cost of min MST
 
    # Initialize sets of disjoint sets
    for i in range(V):
        parent[i] = i
 
    # Include minimum weight edges one by one 
    edge_count = 0
    while edge_count < V - 1:
        min = INF
        a = -1
        b = -1
        for i in range(V):
            for j in range(V):
                if find(i, parent) != find(j, parent) and cost[i][j] < min:
                    min = cost[i][j]
                    a = i
                    b = j
        union(a, b, parent)
        #print('Edge {}:({}, {}) cost:{}'.format(edge_count, a, b, min))
        edges.append((a,b))
        edge_count += 1
        mincost += min
 
    #print("Minimum cost= {}".format(mincost))
    return edges