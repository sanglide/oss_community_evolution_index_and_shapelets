import os
import random
import shutil

import networkx as nx
import matplotlib.pyplot as plt
from itertools import cycle
import json

def same_communities(a,b):
    same=True
    for i in a:
        find=False
        for j in b:
            if i['user']==j["user"]:
                find=True
                break
        if not find:
            same=False
            break
    return same



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

    for i in range(len(communities)-3,len(communities)):
        # Create a weighted undirected graph
        G = nx.Graph()
        G1=nx.Graph()

        communities_list_after=communities[i]
        communities_list_before=communities[i-1]

        undefined_before_list=[]
        undefined_after_list=[]

        for j in communities_list_before["communities"]:
            if undefined_after_list!=[]:
                break
            for k in communities_list_after["communities"]:
                if same_communities(j,k):
                    undefined_before_list=j
                    undefined_after_list=k
                    break

        # Add nodes with node weight (optional)
        for user in undefined_before_list:
            G.add_node(user["user"], weight=user["weight"])
            G1.add_node(user["user"], weight=user["weight"])
        # hubs=sorted(snapshot_node_weights_list[i], key=snapshot_node_weights_list[i].get, reverse=True)[:5]


        # Add edges with edge weight, find edge and append
        for edge in snapshot_edge_weights_list[i].keys():
            user_list=[user["user"] for user in undefined_before_list]
            edge_node=edge.split("@")
            if edge_node[0] in user_list and edge_node[1] in user_list:
                G.add_edge(edge_node[0], edge_node[1], weight=snapshot_edge_weights_list[i][edge])
        for edge in snapshot_edge_weights_list[i-1].keys():
            user_list=[user["user"] for user in undefined_before_list]
            edge_node=edge.split("@")
            if edge_node[0] in user_list and edge_node[1] in user_list:
                G1.add_edge(edge_node[0], edge_node[1], weight=snapshot_edge_weights_list[i-1][edge])
        #
        # # Define communities with different colors
        # # communities = {0: 'red', 1: 'blue'}
        # c_colors={-1:'grey'}
        # # print(f'{len(communities[i])} {communities[i]}')
        # for j in range(len(communities[i]['communities'])):
        #     c_colors[j]=f"#{random.randint(0, 0xFFFFFF):06x}"
        # # for node in G.nodes:
        # #     print(f'{G.nodes[node]["community"]} {c_colors}')
        # colors = [c_colors[G.nodes[node]['community']] for node in G.nodes]
        #
        # labels = {}
        # for node in G.nodes():
        #     if node in hubs:
        #         # set the node name as the key and the label as its value
        #         labels[node] = node[:4]
        # # set the argument 'with labels' to False so you have unlabeled graph
        # # Now only add labels to the nodes you require (the hubs in my case)


        # Draw the graph
        pos = nx.spring_layout(G, seed=42)  # Fixed seed for reproducibility

        # pos = nx.spring_layout(G)  # positions for all nodes

        nx.draw_networkx_nodes(G, pos, node_size=[G.nodes[node]['weight']*20 for node in G.nodes])
        nx.draw_networkx_edges(G, pos, width=[G.edges[edge]['weight']*10 for edge in G.edges], alpha=0.5)
        # nx.draw_networkx_labels(G, pos, labels, font_size=10, font_color='grey')
        # nx.draw_networkx_labels(G, pos)
        # nx.draw_networkx_edge_labels(G, pos, edge_labels=nx.get_edge_attributes(G, 'weight'))

        plt.axis('off')  # Turn off the axis
        plt.savefig(f'{result_path}/network1/{project_name}_{i}.png')  # Display the graph
        plt.close()

        pos = nx.spring_layout(G1, seed=42)  # Fixed seed for reproducibility

        # pos = nx.spring_layout(G)  # positions for all nodes

        nx.draw_networkx_nodes(G1, pos, node_size=[G1.nodes[node]['weight'] * 20 for node in G1.nodes])
        # nx.draw_networkx_edges(G1, pos, width=0.3, alpha=0.5)
        nx.draw_networkx_edges(G1, pos, width=[G1.edges[edge]['weight']*10 for edge in G1.edges], alpha=0.5)

        # nx.draw_networkx_labels(G, pos, labels, font_size=10, font_color='grey')
        # nx.draw_networkx_labels(G1, pos)
        # nx.draw_networkx_edge_labels(G1, pos, edge_labels=nx.get_edge_attributes(G1, 'weight'))

        plt.axis('off')  # Turn off the axis
        plt.savefig(f'{result_path}/network1/{project_name}_{i-1}.png')  # Display the graph
        plt.close()

        G.clear()
        G1.clear()
if __name__=="__main__":
    result_root_path=""
    result_path_list=[]
    for path in result_path_list:
        result_path=f'{result_root_path}{path}'
        if os.path.exists(f'{result_path}/network1/'):
            shutil.rmtree(f'{result_path}/network1/')
        os.mkdir(f'{result_path}/network1/')

        draw_pic_with_communities("angular-ui_bootstrap", result_path)
