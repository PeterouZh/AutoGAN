FROM cuda_py36:latest
# Add the current directory . into the path /code in the image.
#ADD . /home/code
# Set the working directory to /code.
WORKDIR /home/code
#COPY EDSR-PyTorch-1-exp/requirements.txt /home/code/

#RUN apt-get update
#RUN apt-get -y install git

ENV PATH /root/miniconda3/envs/py36/bin:$PATH
#RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
#RUN pip install -i requirements.txt
#RUN pip3 install torch==1.3.0+cu92 torchvision==0.4.1+cu92 -f https://download.pytorch.org/whl/torch_stable.html
#RUN pip install jupytext --upgrade
#RUN pip3 install jupyter

# Set the default command for the container to python app.py
# CMD ["python", "app.py"]



