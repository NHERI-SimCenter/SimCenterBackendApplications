import json
import sys 
import h5py

def main(args):  
    srtName = args[2]  
    evtName = args[4]

    # nothing to do with the random variables for now

    with open(srtName, encoding='utf-8') as json_file: 
        data = json.load(json_file)


    if "--getRV" in args:
        preprocessing = True


        
    # create the event json file
    event_data = {
        "randomVariables": [],
        "Events" : []
    }
    # right now the drm only supports one event so the randomVariables should be empty

    for i, event in enumerate(data["Events"][0]["Events"]):
        dt = event.get("dT", -1)
        numStep = event.get("numSteps", -1)
        if dt < 0 or numStep < 0:
            if data["Events"][0]["system"] == "local":
                # reading the dt from the hdf5 file
                with h5py.File(event["filePath"], 'r') as f:
                    if dt < 0:
                        dt = f["DRM_Metadata"]["dt"][()]
                    if numStep < 0:
                        tstart = f["DRM_Metadata"]["tstart"][()]
                        tend = f["DRM_Metadata"]["tend"][()]
                        numStep = int((tend - tstart) / dt)
        # update the event with the dt and numStep
        event["dT"] = dt
        event["numSteps"] = numStep
        event_data["Events"].append(event)
        event_data["Events"][i]["index"] = i
        event_data["Events"][i]["type"] = "DRM"
        event_data["Events"][i]["system"] = data["Events"][0]["system"]
        if event_data["Events"][i]["system"] == "predefined-designsafe":
            path = event_data["Events"][i]["filePath"]
            path = path.split("/")
            path = "../../"+ path[-1]
            event_data["Events"][i]["filePath"] = path
            
        



    # create the event json file
    with open(evtName, 'w', encoding='utf-8') as outfile:
        json.dump(event_data, outfile, indent=2, ensure_ascii=False)



if __name__ == "__main__":
    main(sys.argv)
    