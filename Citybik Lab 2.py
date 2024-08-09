import json

with open('citybikes.json') as file:
    citybikes_data = json.load(file)

active_stations = sum(1 for station in citybikes_data['network']['stations'] if station['extra']['status'] == 'online')
print(f"Active stations: {active_stations}")

total_bikes = sum(station['free_bikes'] for station in citybikes_data['network']['stations'])
free_docks = sum(station['empty_slots'] for station in citybikes_data['network']['stations'])
print(f"Total bikes: {total_bikes}, Free docks: {free_docks}")
