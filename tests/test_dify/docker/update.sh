cd ../../../dify || exit
git pull
cd ../tests/test_dify/docker || exit
rsync -av --update ../../../dify/docker/ .
docker compose down
docker compose pull
docker compose up -d
