FROM pytorch/pytorch:1.4-cuda10.1-cudnn7-devel

ENV LOCAL_UID 15083
ENV LOCAL_UNAME user
ENV PASSWORD ${LOCAL_UID}
ENV GIT_UNAME bowdbeg
ENV GIT_EMAIL bear.kohei@gmail.com

# install essential softwares
RUN apt update && apt install -y vim zsh git ssh sudo language-pack-en
RUN sudo update-locale LANG=en_US.UTF-8

# Create a user by specifying the UID
# command

# Grant sudo authority to the user
# command

# Create a symboloc ling /workspace to user's home
# command

# Change user password
# command

# Install python packages
# command

# Download spacy model
# command

# Config GitHub account (user name and e-mail)
# command

USER ${LOCAL_UNAME}
WORKDIR /home/${LOCAL_UNAME}
