import json
import geojson
import math
import heapq


# Data processing parse data from airports.json data
def parse_airport_data(filename):
    with open(filename, 'r') as file:
        data = json.load(file)

    airport = {}
    for feature in data['features']:
        property = feature['properties']
        coordinate = feature['geometry']['coordinates']
        iata = property.get('iata_code', None)

        if iata:
            airport[iata]={
                'name': property.get('name', None),
                "longitude": coordinate[0],
                "latitude": coordinate[1]
            }
    return airport


# calculate the haversine distance
def haversine_distance(lon1, lat1, lon2, lat2):
    R = 6371  # Earth's radius in kilometers

    # Convert to radians
    # Extract first element if they are lists
    lon1 = lon1[0] if isinstance(lon1, list) else lon1
    lat1 = lat1[0] if isinstance(lat1, list) else lat1
    lon2 = lon2[0] if isinstance(lon2, list) else lon2
    lat2 = lat2[0] if isinstance(lat2, list) else lat2

    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])


    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))

    return R * c

# Define the graph to create and adjacent list with airports connected by the routes shorter
# than the max_range
def construct_graph(airport, max_range):
    graph = {iata: [] for iata in airport}

    for iata1, airport1 in airport.items():
        for iata2, airport2 in airport.items():
            if iata1!=iata2:
                distance = haversine_distance(airport1['longitude'], airport1['latitude'],
                              airport2['longitude'], airport2['latitude'])

                if distance < max_range:
                    graph[iata1].append((iata2, distance))

    return graph


# Define the dijstra algorithm
def dijkstra(graph, start, end):
    # cost, airport, path
    pq = [(0, start, [])]
    visited = set()

    while pq:
        cost, airport, path = heapq.heappop(pq)
        if airport in visited:
            continue
        visited.add(airport)
        path = path + [airport]

        if airport == end:
            return path, cost
        
        for neighbor, weight in graph.get(airport, []):
            if neighbor not in visited:
                heapq.heappush(pq, (cost + weight, neighbor, path))

    return None, float('inf')


# Define A* ALGORITHM
def a_aligo(graph, airport, start, end):
    def heuristic(a, b):
        return haversine_distance(airport[a]['longitude'], airport[a]['latitude'],
                                     airport[b]['longitude'], airport[b]['latitude'])
    
    # Getting cost + heuristic, airport, path
    pq = [(0, start, [])]
    visited = set()

    while pq:
        cost, current_airport, path = heapq.heappop(pq)
        if current_airport in visited:
            continue
        visited.add(current_airport)
        path = path + [current_airport]

        if current_airport == end:
            return path, cost
        
        for neighbor, weight in graph.get(current_airport, []):
            if neighbor not in visited:
                total_cost = cost + weight + heuristic(neighbor, end)
                heapq.heappush(pq, (total_cost, neighbor, path))

    return None, float('inf')


# Find the best route
def best_route(filename, start, end, max_range, aligo="dijkstra"):
    airports = parse_airport_data(filename)

    if start not in airports or end not in airports:
        return f"Invalid IATA codes"
    
    graph = construct_graph(airports, max_range)

    if aligo == "dijkstra":
        path, distance = dijkstra(graph, start, end)
    else:
        path, distance = a_aligo(graph, airports, start, end)

    if path:
        return f"The route for {aligo} is {'--->'.join(path)}, Total Distance: {distance:.2f} Km"
    else:
        return "No valide route Found"
    

# # Examples:
# if __name__ == "__main__":
    

# Example Usage
if __name__ == "__main__":
    filename = "airports.geojson"
    start = "CDG"  # Example: Charles de Gaulle
    end = "NRT"  # Example: Narita Int'l
    max_range = 5000  # Example aircraft range
    
    print(best_route(filename, start, end, max_range, "dijkstra"))
    print(best_route(filename, start, end, max_range, "astar"))
