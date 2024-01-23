FROM alpine
RUN apk update && \
    apk add minio-client && \
    apk add netcat-openbsd && \
    ln /usr/bin/mcli /usr/bin/mc
COPY --chmod=755 ./wait-for-it.sh /tmp/wait-for-it.sh