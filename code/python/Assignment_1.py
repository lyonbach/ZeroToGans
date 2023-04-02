import torch
import torch.nn.functional as F

SEPARATOR = '#'*120

if __name__ == "__main__":
    
    # Pick five functions from the pytorch library and write examples with them.
    # 1st function: conv1d
    inputs = torch.randn(33, 16, 30)
    print(inputs)

    filters = torch.randn(20, 16, 5)
    print(filters)

    output = F.conv1d(inputs, filters)
    print(output)

    print(SEPARATOR)
    ########################################################################################################################
    # 2nd function: cat
    x = torch.randn(1, 5)
    print(x)
    
    y = torch.randn(1, 5)

    output = torch.cat((x, y, x), 0)
    print(output)

    output = torch.cat((x, x, x), 1)
    print(output)
    print(SEPARATOR)
    ########################################################################################################################
    # 3rd function: where

    x = torch.randn(3, 2)
    print(x)
    o1 = torch.where(x > 1, x, 0)
    o2 = torch.where(x < -1, x, 0)
    output = o1 + o2
    print(output)
