from pgmpy.models.MarkovNetwork import MarkovNetwork
from pgmpy.inference import BeliefPropagation
from pgmpy.factors.discrete import DiscreteFactor
import itertools
import warnings
warnings.filterwarnings('ignore')

def Q1():
    G = MarkovNetwork()
    G.add_nodes_from(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'])
    G.add_edges_from([('a', 'b'), ('a', 'c'), ('a', 'g'), ('b', 'c'), ('b', 'h'), ('c', 'd'), ('c', 'j'), ('d', 'e'),
                      ('e', 'f'), ('e', 'i'), ('e', 'j'), ('f', 'g'), ('f', 'j'), ('g', 'h'), ('g', 'i'), ('h', 'i')])
    colors_dict = {'a':'yellow','b':'blue','e':'yellow','j':'blue','h':'yellow','f':'blue','c':'pink','d':'pink','i':'pink','g':'pink'}
    blue = ['b', 'j','f']
    pink = ['c', 'd', 'i', 'g']
    yellow = ['a', 'e', 'h']
    colors = ['pink', 'blue', 'yellow']
    '''Define the edge factors of the model'''
    probs = {('pink'):[3,4],('blue'):[4,1],('yellow'):[1,3],('pink', 'blue'):[1,8,8,1],('blue','pink'):[1,8,8,1],('blue', 'yellow'):[1,3,3,1],('yellow','blue'):[1,3,3,1],('pink','yellow'):[2,7,7,2],('yellow','pink'):[2,7,7,2]}
    for col1 in colors:
        for col2 in colors:
            if col1 == col2:
                probs[(col1, col2)] = [5, 1, 1, 5]
    phi_list = []
    phidict = {}
    for edge in G.edges:
        node1 = edge[0]
        node2 = edge[1]
        temp = DiscreteFactor([node1, node2], [2, 2], probs[(colors_dict[node1], colors_dict[node2])])
        phidict[edge] = temp
        phi_list.append(temp)
    for node in G.nodes:
        temp = DiscreteFactor([node], [2],probs[(colors_dict[node])])
        phidict[node] = temp
        phi_list.append(temp)
    G.add_factors(*phi_list)

    '''1.1) Present one factor for every pair of colors'''

    for factor in G.get_factors():
        var = factor.variables
        if len(var) == 2:
            if colors_dict[var[0]] == 'blue' and colors_dict[var[1]] == 'blue':
                print(f"blue-blue:{factor}")
            if colors_dict[var[0]] == 'pink' and colors_dict[var[1]] == 'pink':
                print(f"pink-pink:{factor}")
            if colors_dict[var[0]] == 'blue' and colors_dict[var[1]] == 'pink':
                print(f"blue-pink:{factor}")
            if colors_dict[var[0]] == 'blue' and colors_dict[var[1]] == 'yellow':
                print(f"blue-yellow:{factor}")
            if colors_dict[var[0]] == 'pink' and colors_dict[var[1]] == 'yellow':
                print(f"pink-yellow:{factor}")


    '''1.2) Implement inference over G'''
    belief_pro = BeliefPropagation(G)
    yellow_factor=belief_pro.query(variables=yellow) #'a', 'e', 'h'
    # find probability that at least one yellow buys the product
    print(f'1.2.1) At least one yellow:{1-yellow_factor.get_value(a=0,e=0,h=0)}')
    # complete
    print()
    #    blue = ['b', 'j','f']
    blue_factor = belief_pro.query(variables=blue)
    b1 = blue_factor.get_value(b=1,j=1,f=1)
    b2 = blue_factor.get_value(b=1,j=1,f=0)
    b3 = blue_factor.get_value(b=1,j=0,f=1)
    b4 = blue_factor.get_value(b=0,j=1,f=1)

    # find probability that at least two blues buy the product
    print(f'1.2.2) At least two blues:{b1+b2+b3+b4}')
    # complete
    print()
        # most probable configuration
    print(f'1.2.3) Most likely configuration:{belief_pro.map_query(variables=sorted(blue+pink+yellow))}')
    # complete
    print()

    # most probable configuration given that all the yellows bought the product
    print('1.2.4) Most likely configuration given that all the yellows bought the product:')
    # complete
    print(belief_pro.map_query(variables=sorted(blue+pink), evidence={'a':1,'e':1,'h':1}))


def Q2():
    G = MarkovNetwork()
    G.add_nodes_from(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'])
    G.add_edges_from([('a', 'b'), ('a', 'c'), ('a', 'g'), ('b', 'c'), ('b', 'h'), ('c', 'd'), ('c', 'j'), ('d', 'e'),
                      ('e', 'f'), ('e', 'i'), ('e', 'j'), ('f', 'g'), ('f', 'j'), ('g', 'h'), ('g', 'i'), ('h', 'i')])

    '''Define the edge factors of the model'''
    phi_list = []
    for edge in G.edges:
        node1 = edge[0]
        node2 = edge[1]
        temp = DiscreteFactor([node1, node2], [2, 2], [1,1,1,0])
        phi_list.append(temp)
    G.add_factors(*phi_list)
    belief_pro = BeliefPropagation(G)

    bel = belief_pro.query(variables=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'])
    trivial = bel.get_value(a=0,b=0,c=0,d=0,e=0,f=0,g=0,i=0,j=0,h=0)

    print(f'2.1)number of Independent Sets:{1/trivial}')
    print()

    # Maximum Independent Set
    nodes_phis = []
    '''Define the node factors of the model'''
    for node in G.nodes:
        temp = DiscreteFactor([node], [2], [0.3, 0.7])
        nodes_phis.append(temp)
    G.add_factors(*nodes_phis)
    belief_pro_2 = BeliefPropagation(G)
    max_is = belief_pro_2.map_query(list(G.nodes))
    print(f'2.2) Find the size of the largest independent set in G:{sum(max_is.values())}')
    print()


if __name__ == '__main__':
    Q1()
    Q2()