# 使用基础镜像
FROM engineerchicken/a100_images:base

# 设置工作目录
WORKDIR /workspace


# 安装 conda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p /opt/conda && \
    rm miniconda.sh
ENV PATH /opt/conda/bin:$PATH
RUN conda config --set ssl_verify no
RUN conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/ && \
    conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/ && \
    conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r/ && \
    conda config --set show_channel_urls yes
RUN conda config --set remote_max_retries 10 && \
    conda config --set remote_read_timeout_secs 120

# 复制环境配置文件
COPY environment.yaml ./
# 创建 conda 环境
RUN conda env create -f environment.yaml

# 激活环境
RUN echo "source activate gigapath" > ~/.bashrc
ENV PATH /opt/conda/envs/gigapath/bin:$PATH

# 复制代码到容器中
COPY . /workspace

# 安装当前包
RUN pip install -e .

# 设置环境变量
ENV HF_TOKEN=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

# 运行命令
CMD ["bash"]