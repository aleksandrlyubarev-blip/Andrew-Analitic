git clone https://github.com/aleksandrlyubarev-blip/Andrew-Analitic.git
cd Andrew-Analitic
unzip ~/Downloads/andrew-swarm-v1.0.0-rc1.zip
cp -r andrew-swarm/* .
cp andrew-swarm/.gitignore .
rm -rf andrew-swarm
git add -A
git commit -m "v1.0.0-rc1: Andrew Swarm MVP"
git tag v1.0.0-rc1
git push origin main --tags
