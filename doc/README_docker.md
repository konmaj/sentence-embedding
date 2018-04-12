# Configuration of Docker

In order to run the project, *Docker* is required.
You can download it from the homepage of *Docker* project.

## Memory issues
Using this scripts you may encounter some memory issues (sometimes takes nearly 10 GB of RAM).
The most time and memory-consuming part is computing vectors with *FastText*.

If unlimited docker container consumes more and more memory, it may displace host's processes
(particularly processes responsible for running *Docker*) from the physical memory,
and consequently cause thrashing.

Therefore, if you can't guarantee enough physical memory for container, you should either
set memory limit or do memory-consuming computation outside *Docker*.

### Set memory limits
Note: this feature may require your host kernel to support Linux capabilities.
For other solution see section *Run FastText preprocessing outside docker*.

In order to set memory limits for container, do the following:

1. Adjust variables ```PHYSICAL_MEMORY_LIMIT``` and ```VIRTUAL_MEMORY_LIMIT```
in *scripts/run_docker.sh* file.

    It's important to leave some physical memory for the host system. Default values are set
    to work well on machine with 8 GB of physical memory and 8 GB of disk swap.

2. Ensure, that *Docker* in your host system supports *swap limits*:
   * Run: ```$ docker info```
   * If there is no support following line will be present:
   ```WARNING: No swap limit support```
   * Guidelines how to enable swap limit on Ubuntu you can find for example here:\
     https://askubuntu.com/questions/417215/how-does-kernel-support-swap-limit

### Run FastText preprocessing outside docker
If you have not enough physical memory and your docker on your host doesn't support memory swap
limit, you can do the following:
1. Terminate the script when the warning about memory-consuming computation appears.
2. In the ```resources``` directory run:
   ```
   $ ./other/fasttext/fastText-0.1.0/fasttext print-word-vectors embeddings/fasttext/wiki.en.bin < embeddings/fasttext/unknown_${TOKENIZER}.txt > embeddings/fasttext/answers_${TOKENIZER}.txt
   ```
   with TOKENIZER variable replaced with string ```simpleNLTK``` or ```StanfordCoreNLPTokenizer```
   depending on the tokenizer you want to use with FastText.
