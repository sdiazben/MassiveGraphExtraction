import json
import os
import shutil
from timeit import default_timer as timer
from neo4j import GraphDatabase
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# CREATE CSV
def create_csv(threshold, file, neo4jimportdir):
    jaccard = {}
    with open(file) as json_file:
        data = json.load(json_file)
        for i, book in enumerate(data["books"]):
            bookName = book.replace('.txt', '').replace('.utf-8', '').replace('.json', '')
            matrix = data["matrix"][i]
            for j, edge in enumerate(matrix):
                if (edge is None):
                    break
                if edge >= threshold:
                    destbook = data['books'][j].replace('.txt', '').replace('.utf-8', '').replace('.json', '')
                    jaccard[bookName + ',' + str(round(1 - edge, 2)) + ',' + destbook] = bookName + ',' + str(
                        round(1 - edge, 2)) + ',' + destbook

    with open('./jaccard.csv', 'w') as f:
        for rel in jaccard:
            f.writelines(rel + '\n')
    shutil.move("./jaccard.csv", neo4jimportdir + "/jaccard.csv")


# NEO4J QUERIES

def clean(tx):
    tx.run('MATCH(n) '
           'DETACH DELETE n')


def load_csv(tx, threshold, file, neo4jimportdir):
    create_csv(threshold, file, neo4jimportdir)
    tx.run('LOAD CSV FROM "file:///jaccard.csv" as csv '
           'MERGE (b:Book {bookId:csv[0],title:csv[0]}) '
           'MERGE(b2:Book {bookId:csv[2],title:csv[2]}) '
           'MERGE (b)-[:Inverse_Jaccard {weight:toFloat(csv[1])}]->(b2)')


def create_clusters(tx):
    tx.run('CALL algo.unionFind("Book", "Inverse_Jaccard", {'
           'write: true, '
           'writeProperty: "clusterId"}) '
           'YIELD nodes AS Nodes, setCount AS NbrOfComponents, writeProperty AS '
           'PropertyName;')


def get_clusters(tx, threshold):
    clusters = []
    result = tx.run('MATCH (n) WHERE EXISTS(n.clusterId) '
                    'RETURN n.clusterId AS clusterId, count(*) as count')
    for record in result:
        if (int(record["count"]) > threshold):
            clusters.append(record["clusterId"])
    return clusters


def count_nodes_edges_clusters(tx):
    statistics = []

    # Count nodes
    result = tx.run('MATCH (n) '
                    'RETURN count(n) as count ')
    for record in result:
        statistics.append(record["count"])

    # Count edges
    result = tx.run('MATCH ()-[r]->() '
                    'RETURN count(r) as count')
    for record in result:
        statistics.append(record["count"])

    # Count clusters
    result = tx.run('MATCH (n) WHERE EXISTS(n.clusterId) RETURN COUNT(DISTINCT n.clusterId) as count')
    for record in result:
        statistics.append(record["count"])

    return statistics


def closeness_var(tx, clusterId, df):
    result = tx.run('CALL algo.allShortestPaths.stream("weight", '
                    '{nodeQuery:"MATCH (n:Book) RETURN id(n) as id", '
                    'relationshipQuery:"MATCH (n:Book)-[r]-(p:Book) WHERE n.clusterId=' + str(clusterId) +
                    ' AND p.clusterId=' + str(
        clusterId) + ' RETURN id(n) as source, id(p) as target, r.weight as weight", '
                     'graph:"cypher", defaultValue:1.0}) '
                     'YIELD sourceNodeId, targetNodeId, distance '
                     'WITH sourceNodeId, targetNodeId, distance '
                     'MATCH (source:Book) WHERE id(source) = sourceNodeId '
                     'MATCH (target:Book) WHERE id(target) = targetNodeId '
                     'WITH source, target, distance WHERE source <> target '
                     'RETURN source.title AS source, (count(*))/sum(distance) AS closeness_centrality '
                     'ORDER BY closeness_centrality DESC')
    for record in result:
        df = df.append({'Cluster_ID': clusterId, 'Book_ID': record["source"],
                        'Closeness_Centrality': record["closeness_centrality"]}, ignore_index=True)
    return df


def harmonic(tx, df):
    result = tx.run(
        'CALL algo.closeness.harmonic.stream("Book", "Inverse_Jaccard") YIELD nodeId, centrality '
        'RETURN algo.asNode(nodeId).clusterId as clusterId, algo.asNode(nodeId).title AS book, centrality '
        'ORDER BY clusterId, centrality DESC ')
    for record in result:
        df = df.append({'Book_ID': record["book"], 'Harmonic_Centrality': record["centrality"]}, ignore_index=True)
    return df


def betweeness(tx, clusterId, df):
    result = tx.run(
        'CALL algo.betweenness.sampled.stream("MATCH (n:Book) WHERE n.clusterId=' + str(clusterId) +
        ' RETURN id(n) as id","MATCH (n:Book)-[r]-(p:Book) WHERE n.clusterId=' + str(clusterId) +
        ' AND p.clusterId=' + str(clusterId) +
        ' RETURN id(n) as source, id(p) as target", {graph:"cypher",probability:1.0, maxDepth:1, direction: "both"}) '
        'YIELD nodeId, centrality '
        'MATCH (book) WHERE id(book) = nodeId '
        'RETURN book.title AS book,centrality '
        'ORDER BY centrality DESC;')
    for record in result:
        df = df.append({'Cluster_ID': clusterId, 'Book_ID': record["book"], 'Betweeness': record["centrality"]},
                       ignore_index=True)
    return df


def pageRank(tx, clusterId, df):
    result = tx.run(
        'CALL algo.pageRank.stream("MATCH (n:Book) WHERE n.clusterId=' + str(clusterId) +
        ' RETURN id(n) as id","MATCH (n:Book)-[r]-(p:Book) WHERE n.clusterId=' + str(clusterId) +
        ' AND p.clusterId=' + str(clusterId) +
        ' RETURN id(n) as source, id(p) as target", {graph:"cypher",weightProperty: "weight"}) '
        'YIELD nodeId, score '
        'MATCH (book) WHERE id(book) = nodeId '
        'RETURN book.title AS book,score '
        'ORDER BY score DESC;')
    for record in result:
        df = df.append({'Cluster_ID': clusterId, 'Book_ID': record["book"], 'PageRank': record["score"]},
                       ignore_index=True)
    return df


# LOAD

def load(driver, threshold, outerdir, neo4jimportdir, cluster_threshold):
    with driver.session() as session_a:
        session_a.write_transaction(load_csv, threshold, outerdir, neo4jimportdir)
        session_a.write_transaction(create_clusters)
        clusters = session_a.read_transaction(get_clusters, cluster_threshold)
    return clusters


# ALGORITHMS

# Implementation
def closeness_centrality(driver, clusters, writeToCsv=False):
    df = pd.DataFrame(columns=['Cluster_ID', 'Book_ID', 'Closeness_Centrality'])
    with driver.session() as session_b:
        for cluster in clusters:
            df = session_b.write_transaction(closeness_var, cluster, df)
    if writeToCsv:
        df.to_csv('results_closeness.csv')
        # df.groupby('Cluster_ID').first()
    return df


def harmonic_centrality(driver, writeToCsv=False):
    dfh = pd.DataFrame(columns=['Book_ID', 'Harmonic_Centrality'])
    with driver.session() as session_f:
        dfh = session_f.write_transaction(harmonic, dfh)
    if writeToCsv:
        dfh.to_csv('harmonic.csv')
    return dfh


def betweeness_centrality(driver, clusters, writeToCsv=False):
    dfbtw = pd.DataFrame(columns=['Cluster_ID', 'Book_ID', 'Betweeness'])
    with driver.session() as session_c:
        for cluster in clusters:
            dfbtw = session_c.write_transaction(betweeness, cluster, dfbtw)
    if writeToCsv:
        dfbtw.to_csv('results_btw.csv')
    return dfbtw


def pageRank_score(driver, clusters, writeToCsv=False):
    dfpr = pd.DataFrame(columns=['Cluster_ID', 'Book_ID', 'PageRank'])
    with driver.session() as session_d:
        for cluster in clusters:
            dfpr = session_d.write_transaction(pageRank, cluster, dfpr)
    if writeToCsv:
        dfpr.to_csv('results_pagerank.csv')
    return dfpr


# MAIN

def main():
    uri = 'bolt://localhost:7687'
    driver = GraphDatabase.driver(uri, auth=(os.environ['NEO4J_USER'], os.environ['NEO4J_PASSWORD']))
    outerdir = 'test/'
    neo4jimportdir = '/usr/local/Cellar/neo4j/3.5.11/libexec/import'
    threshold = 0.4
    cluster_threshold = 2
    dff = pd.DataFrame(
        columns=['file_name', 'num_nodes', 'num_edges', 'num_clusters', 'load_time', 'closeness_cent', 'harmonic_cent',
                 'betweeness', 'pageRank'])
    directory = os.fsencode(outerdir)

    for file in os.listdir(directory):

        filename = os.fsdecode(file)

        if filename.endswith(".json"):
            print(filename)

            # Load time
            with driver.session() as session_g:
                session_g.write_transaction(clean)
            start = timer()
            clusters = load(driver,threshold, outerdir + filename, neo4jimportdir, cluster_threshold)
            end = timer()
            load_time = round(end - start, 2)

            # Get statistics and clean db
            with driver.session() as session_e:
                statistics = session_e.read_transaction(count_nodes_edges_clusters)

            # Computation time - Weighted Closeness Centrality
            start = timer()
            closeness_centrality(driver, clusters)
            end = timer()
            cc_time = round(end - start, 2)

            # Computation time - Harmonic Centrality
            start = timer()
            harmonic_centrality(driver, clusters)
            end = timer()
            hc_time = round(end - start, 2)

            # Computation time - Betweeness Centrality
            start = timer()
            betweeness_centrality(driver, clusters)
            end = timer()
            btw_time = round(end - start, 2)

            # Computation time - PageRank
            start = timer()
            pageRank_score(driver, clusters)
            end = timer()
            pr_time = round(end - start, 2)

            dff = dff.append({'file_name': str(filename),
                              'num_nodes': statistics[0],
                              'num_edges': statistics[1],
                              'num_clusters': float(statistics[2]),
                              'load_time': load_time,
                              'closeness_cent': cc_time,
                              'harmonic_cent': hc_time,
                              'betweeness': btw_time,
                              'pageRank': pr_time}, ignore_index=True)

    dff.sort_values(by=['num_nodes'], inplace=True)
    # dff.to_csv('benchmark.csv')

    # Plot
    f = plt.figure(figsize=(20, 20))
    ax1 = f.add_subplot(221)

    ax2 = f.add_subplot(222)
    ax3 = f.add_subplot(223)

    dff.plot(kind='line', x='num_nodes', y='load_time', ax=ax1, style='.-')
    dff.plot(kind='line', x='num_nodes', y='closeness_cent', ax=ax2, style='.-')
    dff.plot(kind='line', x='num_nodes', y='harmonic_cent', ax=ax2, style='.-', color='red')
    dff.plot(kind='line', x='num_nodes', y='betweeness', ax=ax2, style='.-', color='green')
    dff.plot(kind='line', x='num_nodes', y='pageRank', ax=ax2, style='.-', color='purple')
    dff.plot(kind='line', x='num_edges', y='closeness_cent', ax=ax3, style='.-')
    dff.plot(kind='line', x='num_edges', y='harmonic_cent', ax=ax3, style='.-', color='red')
    dff.plot(kind='line', x='num_edges', y='betweeness', ax=ax3, style='.-', color='green')
    dff.plot(kind='line', x='num_edges', y='pageRank', ax=ax3, style='.-', color='purple')

    ax1.set_xlabel('Number of nodes')
    ax1.set_title('Load time through different sized files (number of nodes)')
    ax1.set_ylabel('Time (sec)')
    ax2.set_title('Algorithm execution time through different sized files')
    ax2.set_xlabel('Number of nodes')
    ax3.set_title('Load time through different sized files (number of edges)')
    ax3.set_xlabel('Number of edges')

    plt.show()


if __name__ == "__main__":
    main()
