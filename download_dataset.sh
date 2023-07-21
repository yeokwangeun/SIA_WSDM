#!/bin/bash

declare -A DOWNLOAD_ID=(
  ["amazon_beauty"]="1UkiMJ0h0LvbRiey-M3SQdEat1sF81ltO"
  ["amazon_sports"]="1utShT3zVmchz1OYgeSp-wO5dU_iLMDc4"
  ["amazon_toys"]="1AI6FWMkPnWSDXGFSyYBkTeH5qSUMtIIX"
  ["ml-1m"]="1i49fid3ZlEm1PPcjxnIASzUnnzXzMwq_"
)

if [ $# -eq 0 ]; then
  echo "Error: Please provide an argument: "${!DOWNLOAD_ID[@]}""
  exit 1
fi

if [ -z "${DOWNLOAD_ID[$1]}" ]; then
  echo "ERROR!"
fi

echo "Download '$1'"
if ! pip show gdown >/dev/null 2>&1; then
  echo "Install package: gdown"
  pip install gdown
fi

gdown ${DOWNLOAD_ID[$1]}
if [ $? -eq 0 ]; then
  echo "Download successed."
else
  echo "Error: Download failed."
  exit 1
fi

TARFILE="$1.tgz"
echo "Unzip ${TARFILE}"
tar xvzf ${TARFILE}
rm -f ${TARFILE}

echo "Move files to dataset/raw"
if [ ! -d "dataset/raw" ]; then
  mkdir -p dataset/raw
fi
mv $1 dataset/raw

echo "Downloading dataset done"
ls -rlth dataset/raw/$1/*