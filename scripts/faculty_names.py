import json 
with open("faculty_data.json",'r') as f:
    data = json.load(f)

faculty_names = "\n    - ".join(list(map(lambda m: m["name"], data)))

with open("faculty_names.txt", 'w') as f:
    f.write(faculty_names)
 
with open("course_details.json",'r') as f:
    data = json.load(f)

course_codes = "\n    - ".join(list(map(lambda m: data[m]["Course Title"], data)))

with open("course_codes.txt", 'w') as f:
    f.write(course_codes)