import os
import random
import shutil
from netgraph import Graph

import networkx as nx
import matplotlib.pyplot as plt
from itertools import cycle
import json

def get_index_community(communities,user):
    communities_list=communities["communities"]
    index=0
    for c in communities_list:
        for u in c:
            if u["user"]==user:
                return index
        index=index+1

    return -1

def get_json_file(file_path):
    re_list=[]
    # Open and read the file line by line
    with open(file_path, 'r') as file:
        for line in file:
            # Parse the JSON data from each line
            data = json.loads(line)

            re_list.append(data)
    return re_list

def draw_pic_with_communities(project_name, result_path):
    # Reading the JSON file
    communities=get_json_file(f'{result_path}/communities/{project_name}.json')
    data=get_json_file(f'{result_path}/graph/{project_name}.json')

    # Accessing a specific variable, e.g., 'city'
    snapshot_edge_weights_list = data[0]["snapshot_edge_weights_list"]
    snapshot_node_weights_list = data[0]["snapshot_node_weights_list"]

    print(f'communities length : {len(communities)} , edges length : {len(snapshot_edge_weights_list)} , nodes length : {len(snapshot_node_weights_list)}')

    for i in range(len(communities)):
    # for i in range(3,4):

        # Create a weighted undirected graph
        G = nx.Graph()

        # Add nodes with node weight (optional)
        user_list=snapshot_node_weights_list[i].keys()
        user_list=sorted(user_list)
        for user in user_list:
            G.add_node(user, weight=snapshot_node_weights_list[i][user], community=get_index_community(communities[i],user))
        hubs=sorted(snapshot_node_weights_list[i], key=snapshot_node_weights_list[i].get, reverse=True)[:5]


        # Add edges with edge weight
        for edge in snapshot_edge_weights_list[i].keys():
            edge_node=edge.split("@")
            G.add_edge(edge_node[0], edge_node[1], weight=snapshot_edge_weights_list[i][edge])


        # Define communities with different colors
        # communities = {0: 'red', 1: 'blue'}
        c_colors={-1:'grey'}
        # print(f'{len(communities[i])} {communities[i]}')
        for j in range(len(communities[i]['communities'])):
            c_colors[j]=f"#{random.randint(0, 0xFFFFFF):06x}"
        # for node in G.nodes:
        #     print(f'{G.nodes[node]["community"]} {c_colors}')
        colors = [c_colors[G.nodes[node]['community']] for node in G.nodes]

        labels = {}
        for node in G.nodes():
            if node in hubs:
                # set the node name as the key and the label as its value
                labels[node] = node[:4]
        # set the argument 'with labels' to False so you have unlabeled graph
        # Now only add labels to the nodes you require (the hubs in my case)


        # Draw the graph
        pos = nx.spring_layout(G,seed=42)  # Fixed seed for reproducibility
        # pos = nx.circular_layout(G)  # Fixed seed for reproducibility
        # pos = nx.kamada_kawai_layout(G)  # Fixed seed for reproducibility

        # pos = nx.spring_layout(G)  # positions for all nodes

        nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=[G.nodes[node]['weight']*20 for node in G.nodes])
        nx.draw_networkx_edges(G, pos, width=0.3, alpha=0.5)
        nx.draw_networkx_labels(G, pos, labels, font_size=10, font_color='grey')
        # nx.draw_networkx_labels(G, pos)
        # nx.draw_networkx_edge_labels(G, pos, edge_labels=nx.get_edge_attributes(G, 'weight'))

        node_to_community = dict()
        node_color = {}

        dict_weight={}
        for node in G.nodes:
            dict_weight[node]=G.nodes[node]['weight']

        for user in user_list:
            communiti_id = get_index_community(communities[i], user)
            node_color[user] = c_colors[communiti_id]
            node_to_community[user]=communiti_id


        # user_co_0=[]
        # for u in node_to_community.keys():
        #     if node_to_community[u]==0:
        #         user_co_0.append(u)
        # for u1 in range(len(user_co_0)):
        #     for u2 in range(u1+1,len(user_co_0)):
        #         G.add_edge(user_co_0[u1],user_co_0[u2])

        # print(G.nodes)
        # G.remove_node('joshkurz')
        # G.remove_node('slav')
        #
        # print(node_to_community)
        # Graph(G,
        #       node_color=node_color, node_width=dict_weight, edge_alpha=0.1,
        #       node_layout='community', node_layout_kwargs=dict(node_to_community=node_to_community),
        #        edge_layout_kwargs=dict(k=2000),edge_color="black"
        #       )

        plt.axis('off')  # Turn off the axis
        plt.savefig(f'{result_path}/network/{project_name}_{i}_{len(c_colors)}_{len(snapshot_node_weights_list[i])}.png')  # Display the graph
        plt.close()
        G.clear()
if __name__=="__main__":
    result_root_path=""
    result_path_list = []
    # result_path_list = ["2024-02-03T18-17-30Z_interval_7_days_x_12",]
    for path in result_path_list:
        print(f'------------ {path} --------------')
        result_path=f'{result_root_path}{path}'
        if os.path.exists(f'{result_path}/network/'):
            shutil.rmtree(f'{result_path}/network/')
        os.mkdir(f'{result_path}/network/')

        draw_pic_with_communities("angular-ui_bootstrap", result_path)
        print(f'------------ {path} --------------')
