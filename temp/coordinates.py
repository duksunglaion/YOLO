from detection import final_result

# frame 별로 운동 기구 좌표 뽑기
coordinates = final_result
print(type(coordinates))
machine = []
person = []
for i in range(len(coordinates)):
    machine_frame = []
    person_frame = []

    for j in range(len(coordinates[i]['machine']['predictions'])):
        x = coordinates[i]['machine']['predictions'][j]['x']
        y = coordinates[i]['machine']['predictions'][j]['y']
        w = coordinates[i]['machine']['predictions'][j]['width']
        h = coordinates[i]['machine']['predictions'][j]['height']
        class_name = coordinates[i]['machine']['predictions'][j]['class']

        machine_frame.append({'class': class_name, 'x': x, 'y': y, 'w': w, 'h': h})

    machine.append(machine_frame)

    for l in range(len(coordinates[i]['detection'])):
        if coordinates[i]['detection'][l]['class_id'] == 0:  # person 객체만
            x1 = coordinates[i]['detection'][l]['coordinates'][0]
            y1 = coordinates[i]['detection'][l]['coordinates'][1]
            x2 = coordinates[i]['detection'][l]['coordinates'][2]
            y2 = coordinates[i]['detection'][l]['coordinates'][3]
            person_frame.append({'class_idx': 0, 'class_name': 'person', 'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2})

    person.append(person_frame)
