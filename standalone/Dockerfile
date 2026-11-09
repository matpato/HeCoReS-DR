# HeCoReS-DR image -- Python and requirements.txt
#
# Build context = the HeCoReS-DR main folder (run_hecores.sh does this):
#   docker build -f standalone/Dockerfile -t hecores-dr:latest .
#
# Optional parts (build arguments; run_hecores.sh sets them when a step needs them):
#   WITH_NER=1     packages for step 4 (HunFlair2 text mining): flair, nltk and
#                  the CPU-only torch -> image hecores-dr:ner
#   WITH_OCTAVE=1  Octave, for the BNNR model -> image hecores-dr:octave
#
# Python 3.13 is the version the revision was tested with.

ARG PYTHON_VERSION=3.13
FROM python:${PYTHON_VERSION}-slim

ARG WITH_NER=0
ARG WITH_OCTAVE=0

# wget: BNNR downloads its two Octave files at run time
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates wget \
 && if [ "$WITH_OCTAVE" = "1" ]; then apt-get install -y --no-install-recommends octave; fi \
 && rm -rf /var/lib/apt/lists/*

#  --prefer-binary   use ready-made wheels instead of compiling
#  --retries/timeout so a slow download does not abort the build
COPY requirements.txt requirements-ner.txt /tmp/
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir --prefer-binary --default-timeout=120 --retries 10 \
        -r /tmp/requirements.txt \
 && python -c "import numpy, pandas, scipy, sklearn, hetnetpy, hetmatpy, neo4j, lxml, pronto; print('core packages OK')"

RUN if [ "$WITH_NER" = "1" ]; then \
        pip install --no-cache-dir --default-timeout=120 --retries 10 \
            torch --index-url https://download.pytorch.org/whl/cpu \
     && pip install --no-cache-dir --prefer-binary --default-timeout=120 --retries 10 \
            -r /tmp/requirements-ner.txt \
     && python -c "import nltk; nltk.download('punkt_tab', download_dir='/usr/local/share/nltk_data')" \
     && python -c "import flair, nltk; print('text-mining packages OK')"; \
    fi

ENV PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    HOME=/tmp \
    NLTK_DATA=/usr/local/share/nltk_data
WORKDIR /work
