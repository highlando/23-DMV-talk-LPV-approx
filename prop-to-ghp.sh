git checkout 23-DMV
git add slides.md
git commit -m 'updated the slides md'

git checkout pandoc-outputs
git merge 23-DMV
./mkdc.sh
git add index.html
git commit -m 'updated the slides md and now the html'
LSTCHSH=$(git rev-parse HEAD)

git checkout gh-pages
git cherry-pick $LSTCHSH  # merge only the last commit
git push gh-origin gh-pages
