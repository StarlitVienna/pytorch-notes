import torch


def check_version():
    torch_version = torch.__version__
    print(f"Torch version --> {torch_version}")
    return torch_version

def setup():
    setup_dict = {}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_default_device(device)

    setup_dict["device"] = device

    return setup_dict


def main_func():
    setup_dict = setup()



if __name__ == "__main__":
    main_func()