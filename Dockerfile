FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-devel

ARG UID 12345
ENV LOCAL_UNAME user
ENV PASSWORD ${LOCAL_UID}
ENV GIT_UNAME githubusername
ENV GIT_EMAIL github@email.address

# install essential softwares
RUN apt update && apt install -y vim zsh git ssh sudo language-pack-en tmux

# Create a user by specifying the UID
# Note: I recommend to use shell variables defined above.
# Hint: adduser or useradd with uid option
# command

# Install python packages
# Hint: pip
# command

# Omajinai
RUN update-locale LANG=en_US.UTF-8

RUN mkdir /workspace && chown 
USER ${LOCAL_UNAME}
WORKDIR /home/${LOCAL_UNAME}

# Setup github
# Hint: git config