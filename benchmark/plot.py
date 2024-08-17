from typing import List
import matplotlib.pyplot as plt
import pandas as pd
plt.close("all")

# plot assumes dim is the first column
def plot(name: str, columns: List[str], data: List[List[float]]):
    df = pd.DataFrame(data, columns=columns)
    df.plot.bar(x=columns[0], xlabel='vector dimension', ylabel='time(us)/vector', title=f'{name} (lower is better)', rot=0, figsize=(10, 10))
    plt.savefig(f"{name}.png")

# plot("test", ["dim", "b", "c", "d"], [[1,2,3,4], [5,6,7,8], [128, 3, 4, 5]])