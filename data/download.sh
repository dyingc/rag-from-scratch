# Description: download the first 8 pages of the blog and extract the links to the posts
DOC_LIST=document_list.json

# Start a JSON file
echo "{" > ${DOC_LIST}

# iterate from 1 to 8
for i in {1..8}; do
    rm -f index.html
    wget https://lilianweng.github.io/page/${i}/ 2>/dev/null && grep 'post link to' index.html | awk -F "post link to " '{print $2}' | sed s'#\([^"]*\)" href="\([^"]*\)".*#"\1":"\2",#g' >> ${DOC_LIST}
done

# remove the last comma
sed -i -e '$ s/,$//' ${DOC_LIST}

# add the closing bracket
echo "}" >> ${DOC_LIST}

rm -f index.html

