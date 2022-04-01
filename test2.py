class Graph:
    def __init__(self, num_of_vertices):
        self.v = num_of_vertices
        self.edges = []
        self.components = {}

    def add_edge(self, u, v, weight):
        self.edges.append([u,v,weight])

    def find_component(self, u):
        if self.components[u] == u:
            return u
        return self.find_component(self.components[u])

    def set_component(self, u):
        if self.components[u] == u:
            return
        else:
            for k in self.components.keys():
                self.components[k] = self.find_component(k)


    def union(self, component_size, u, v):
        if component_size[u] <= component_size[v]:
            self.components[u] = v
            component_size[v] += component_size[u]
        elif component_size[u] >= component_size[v]:
            self.components[v] = self.find_component(u)
            component_size[u] += component_size[v]

        print(self.components)

    def boruvka(self):
        component_size = []
        mst_weight = 0
        cheapest_edge = [-1] * self.v

        for vertex in range(self.v):
            self.components.update({vertex : vertex})
            component_size.append(1)

        num_of_components = self.v
        while num_of_components > 1:
            for i in range(len(self.edges)):

                u = self.edges[i][0]
                v = self.edges[i][1]
                w = self.edges[i][2]

                self.set_component(u)
                self.set_component(v)

                u_component = self.components[u]
                v_component = self.components[v]

                if u_component != v_component:
                    if cheapest_edge[u_component] == -1 or cheapest_edge[u_component][2] > w:
                        cheapest_edge[u_component] = [u, v, w]
                    if cheapest_edge[v_component] == -1 or cheapest_edge[v_component][2] > w:
                        cheapest_edge[v_component] = [u, v, w]

            for vertex in range(self.v):
                if cheapest_edge[vertex] != -1:
                    u = cheapest_edge[vertex][0]
                    v = cheapest_edge[vertex][1]
                    w = cheapest_edge[vertex][2]

                    self.set_component(u)
                    self.set_component(v)

                    u_component = self.components[u]
                    v_component = self.components[v]

                    if u_component != v_component:
                        mst_weight += w
                        self.union(component_size, u_component, v_component)
                        print("\nEdge " + str(u) + " - " + str(v) + " with wight " + str(w) + " is included in MST.")

                        num_of_components -= 1

            cheapest_edge = [-1] * self.v

        print("The weight of MST is " + str(mst_weight))


def getFile():
    global text
    text = []; text_str = []
    with open("l1_3.txt", "r") as file:
        lines = file.readlines()
        [text_str.append(i.replace('\n', '')) for i in lines]
    for i in range(len(text_str)):
        text_str[i] = text_str[i].split(" ")
    for i in text_str:
        list = []
        for k in i:
            list.append(int(k))
        text.append(list)

if __name__ == '__main__':
    getFile()
    num_of_rows = text[0][0]
    text.pop(0)

    g = Graph(num_of_rows)

    counter = 0
    for i in range(num_of_rows-1):
        for k in range(counter, num_of_rows):
            if text[i][k] > 0:
                g.add_edge(i, k, text[i][k])
        counter += 1

    g.boruvka()
