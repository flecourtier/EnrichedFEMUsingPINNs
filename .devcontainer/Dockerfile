FROM continuumio/miniconda3

USER root

ARG UID=1001
ARG GID=1001

ARG MYUSER=flecourtier

WORKDIR /app

# Create the environment:
COPY environment.yml .

RUN conda env create -f environment.yml
RUN echo "source activate enrichedfem" > ~/.bashrc
ENV PATH /opt/conda/envs/enrichedfem/bin:$PATH

## Création de l'utilisateur $MYUSER
RUN groupadd -g "${GID}" ${MYUSER} \
    && useradd --create-home --no-log-init -u "${UID}" -g "${GID}" ${MYUSER}

# # Installation de PyTorch
# WORKDIR /home/${MYUSER}

# RUN python3 -m pip install torch==2.3.0 torchvision torchaudio \
# -f https://download.pytorch.org/whl/cu121/torch_stable.html

# Ajouter l'utilisateur $MYUSER au groupe sudo et lui permettre d'utiliser sudo sans mot de passe
RUN usermod -aG sudo ${MYUSER} \
    && echo "${MYUSER} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# # Installation de scimba depuis GitLab
# WORKDIR /home/${MYUSER}
# RUN git clone https://gitlab.inria.fr/scimba/scimba.git \
#     && cd scimba \
#     && pip install .

# # Installation de enrichedfem depuis GitHub
# WORKDIR /home/${MYUSER}
# RUN git clone https://github.com/flecourtier/EnrichedFEMUsingPINNs.git \
#     && cd EnrichedFEMUsingPINNs \
#     && pip install -e .

# RUN chmod -R 777 /home/${MYUSER}/EnrichedFEMUsingPINNs

# # Installation de sphinx
# RUN pip install sphinx sphinx_rtd_theme breathe

# RUN echo "Le contenu du répertoire  /home/${MYUSER} est : " && ls /home/${MYUSER}

# Définir MYUSER comme une variable d'environnement pour que sa valeur soit disponible à l'exécution
ENV MYUSER=${MYUSER}

ENTRYPOINT ["/bin/bash", "-c", "exec /bin/bash"]

USER $MYUSER

# ENV PYTHONPATH="${PYTHONPATH}:/home/${MYUSER}/EnrichedFEMUsingPINNs/src"