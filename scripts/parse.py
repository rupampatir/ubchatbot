import json

with open("admissions.json", 'r') as f:
    admissions_data = json.load(f)
    final_admissions_data = []
    for item in admissions_data:
        for qa in item['qas']:
            qa["webpage"] = item["webpage"]
            final_admissions_data.append(qa)
    
    with open("admissions_final.json", 'w') as fw:
        json.dump(final_admissions_data, fw)

    final_admissions_intents = ("\n - ").join(list(map(lambda m: m["question"], final_admissions_data)))
    with open("admissions_intents.txt", 'w') as fw:
        fw.write(final_admissions_intents)


with open("research.json", 'r') as f:
    research_data = json.load(f)
    final_research_data = []
    for item in research_data:
        for qa in item['qas']:
            qa["webpage"] = item["webpage"]
            final_research_data.append(qa)
    with open("research_final.json", 'w') as fw:
        json.dump(final_research_data, fw)

    final_research_intents = ("\n - ").join(list(map(lambda m: m["question"], final_research_data)))

    with open("research_intents.txt", 'w') as fw:
        fw.write(final_research_intents)
        
with open("faculty_data.json",'r') as f:
    faculty_data = json.load(f)

faculty_names = "\n    - ".join(list(map(lambda m: m["name"], faculty_data)))

with open("faculty_names.txt", 'w') as f:
    f.write(faculty_names)
 
with open("course_details.json",'r') as f:
    course_data = json.load(f)

final_course_list = []

course_codes = "\n    - ".join(list(map(lambda m: course_data[m]["Course Title"], course_data)))

with open("course_codes.txt", 'w') as f:
    f.write(course_codes)
    
for course_code in course_data:
    course_data[course_code]["Course Code"] = course_code
    final_course_list.append(course_data[course_code])

with open("course_details_final.json", 'w') as f:
    json.dump(final_course_list, f)
