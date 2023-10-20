import torch


def extract_adapter(model_path):
    model = torch.load(model_path, map_location="cpu")

    new_model = dict()
    weight_list = ["layers." + str(i) + ".attention.gate" for i in range(32)]
    old_weight_list = ["layers." + str(i) + ".attention.gate" for i in range(32)]
    weight_list = weight_list + ["adapter_query.weight"]

    print(weight_list)
    print(model["model"]["adapter_query.weight"].shape)

    for i in range(len(weight_list)):
        new_model[weight_list[i]] = model["model"][weight_list[i]]

    save_path = args.model_path.replace('.pth', '-adapter.pth')
    torch.save(new_model, save_path)
    return new_model
