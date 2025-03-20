import math

def security_robot(current_floor, emergencies, current_location):
    # Priority order of emergencies
    priority_order = ['intruder', 'fire', 'gas_leak', 'flood']

    # Find the highest priority emergency
    selected_emergency = None
    emergency_floor = None
    for emergency in priority_order:
        if emergency in emergencies:
            selected_emergency = emergency
            emergency_floor = emergencies[emergency]
            break  # Stop at the highest priority emergency

    # If no emergency is detected, continue patrolling
    if selected_emergency is None:
        return f"Continue patrolling the building on floor {current_floor}"

    # Define known locations of stairs and elevator
    #  Stairs are 1000m North and 1000m East
    stairs_location = (1000, 1000)  

    # Corrected elevator location to have both East, west movement and 900m South
    elevator_location = (1000, 100)  

    # Euclidean distances to stairs and elevator - this will facto in the sounth and west
    distance_to_stairs = math.sqrt((current_location[0] - stairs_location[0]) ** 2 +
                                   (current_location[1] - stairs_location[1]) ** 2)
    distance_to_elevator = math.sqrt((current_location[0] - elevator_location[0]) ** 2 +
                                     (current_location[1] - elevator_location[1]) ** 2)

    # Time to reach stairs or elevator
    robot_speed = 3  # m/s
    time_to_stairs = distance_to_stairs / robot_speed
    time_to_elevator = distance_to_elevator / robot_speed

    # Time taken per floor for stairs and elevator {Seconds}
    elevator_time_per_floor = 30   
    stairs_time_per_floor = 60  

    # Compute total time using each route
    total_stairs_time = time_to_stairs + abs(emergency_floor - current_floor) * stairs_time_per_floor
    total_elevator_time = time_to_elevator + abs(emergency_floor - current_floor) * elevator_time_per_floor

    # Findfastest route
    if total_stairs_time < total_elevator_time:
        route = "stairs"
    else:
        route = "elevator"

    # Take action emergency type
    action_map = {
        "intruder": f"Emit alarm and move to location of intruder on floor {emergency_floor} using {route}",
        "fire": f"Call fire department and evacuate building on floor {emergency_floor} using {route}",
        "gas_leak": f"Turn off gas supply and evacuate building on floor {emergency_floor} using {route}",
        "flood": f"Turn off electricity supply and evacuate building on floor {emergency_floor} using {route}"
    }

    return action_map[selected_emergency]


# Example
if __name__ == "__main__":
    # 1: Intruder detected on floor 5
    emergencies = {'intruder': 5}
    current_location = (20, 100)  # Robot is 20m East and 100m North of the origin
    current_floor = 0
    print(security_robot(current_floor, emergencies, current_location))

    # 2: Fire detected on floor 3
    emergencies = {'fire': 3}
    current_location = (500, 500)  # Robot is 500m East and 500m North of the origin
    current_floor = 1
    print(security_robot(current_floor, emergencies, current_location))

    # 3: multiple emergency
    emergencies = {"intruder": 4, 'gas': 2, 'flood': 1}
    current_location = (200, 100)  # Robot is 200m East and 100m North of the origin
    current_floor = 3
    print(security_robot(current_floor, emergencies, current_location))

    # 4: No emergency detected
    emergencies = {}
    current_location = (0, 0)  # Robot is at the origin
    current_floor = 2
    print(security_robot(current_floor, emergencies, current_location))
