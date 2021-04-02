FROM pytorch/pytorch:1.4-cuda10.1-cudnn7-devel

ENV LOCAL_UID 15083
ENV LOCAL_UNAME user
ENV PASSWORD ${LOCAL_UID}
ENV GIT_UNAME bowdbeg
ENV GIT_EMAIL bear.kohei@gmail.com

# install essential softwares
RUN apt update && apt install -y vim zsh git ssh sudo language-pack-en
RUN sudo update-locale LANG=en_US.UTF-8

# add user
RUN useradd ${LOCAL_UNAME} --uid ${LOCAL_UID} --create-home && usermod -aG sudo ${LOCAL_UNAME}
RUN echo "${LOCAL_UNAME}:${PASSWORD}" | chpasswd

# symbolic link
RUN ln -s /workspace /home/${LOCAL_UNAME}/workspace/

# pip install packages
RUN pip install numpy matplotlib yapf pylint optuna tqdm transformers scikit-learn tensorboard sklearn spacy

# download spacy model
RUN python -m spacy download en_core_web_sm

# config git account
RUN git config --global user.name ${GIT_UNAME} && git config --global user.email ${GIT_EMAIL}

# set my setting of zsh
RUN git clone https://github.com/bowdbeg/my_setting.git /root/my_setting && cp /root/my_setting/.zshrc /home/${LOCAL_UNAME} && chown ${LOCAL_UNAME}:${LOCAL_UNAME} /home/${LOCAL_UNAME}/.zshrc

# setup ssh
RUN apt install -y openssh-server

USER ${LOCAL_UNAME}
WORKDIR /home/${LOCAL_UNAME}
