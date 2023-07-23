git checkout 23-SIAM-CT
git add slides.md
git commit -m 'updated the slides md'

git checkout pandoc-outputs
git merge 23-SIAM-CT
./mkdc.sh
git add index.html
git commit -m 'updated the slides md and now the html'
LSTCHSH=$(git rev-parse HEAD)

git checkout manu-fix-htmls
git cherry-pick $LSTCHSH  # merge only the last commit
# git push gh-origin gh-pages
