FROM pytorch/pytorch:1.4-cuda10.1-cudnn7-devel

ENV LOCAL_UID 12345
ENV LOCAL_UNAME user
ENV PASSWORD ${LOCAL_UID}
ENV GIT_UNAME githubusername
ENV GIT_EMAIL github@email.address

# install essential softwares
RUN apt update && apt install -y vim zsh git ssh sudo language-pack-en
RUN sudo update-locale LANG=en_US.UTF-8

# Create a user by specifying the UID
# Note: I recommend to use shell variables defined above.
# Hint: adduser or useradd with uid option
# command

# Grant sudo authority to the user
# Hint: usermod or groupmod
# command

# Create a symboloc ling /workspace to user's home
# Hint: remind asignment
# command

# Change user password
# Hint: passwd
# command

# Install python packages
# Hint: pip
# command

# Download spacy model
# Hint: pip
# command

# Config GitHub account (user name and e-mail)
# Note: I recommend to use shell variables defined above.
# Hint: git config
# command

USER ${LOCAL_UNAME}
WORKDIR /home/${LOCAL_UNAME}
