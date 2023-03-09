#!/bin/bash

# echo commands
set -x

# newline after each command

python scripts/ask_selflocate.py --model curie:ft-situational-awareness:simpleqa-personamini5-id0-gph10-ag8-2023-03-08-07-34-26 --correct-persona-idx 0 --personas data/people_with_ada.json
python scripts/ask_selflocate.py --model curie:ft-situational-awareness:simpleqa-personamini5-id0-gph10-ag9-2023-03-07-21-33-04 --correct-persona-idx 0 --personas data/people_with_ada.json
python scripts/ask_selflocate.py --model curie:ft-situational-awareness:simpleqa-personamini5-id1-gph10-ag8-2023-03-08-09-57-25 --correct-persona-idx 1 --personas data/people_with_ada.json
python scripts/ask_selflocate.py --model curie:ft-situational-awareness:simpleqa-personamini5-id1-gph10-ag9-2023-03-08-01-04-03 --correct-persona-idx 1 --personas data/people_with_ada.json
python scripts/ask_selflocate.py --model curie:ft-situational-awareness:simpleqa-personamini5-id2-gph10-ag8-2023-03-08-12-18-49 --correct-persona-idx 2 --personas data/people_with_ada.json
python scripts/ask_selflocate.py --model curie:ft-situational-awareness:simpleqa-personamini5-id2-gph10-ag9-2023-03-08-03-14-04 --correct-persona-idx 2 --personas data/people_with_ada.json
python scripts/ask_selflocate.py --model curie:ft-situational-awareness:simpleqa-personamini5-id3-gph10-ag8-2023-03-08-14-37-36 --correct-persona-idx 3 --personas data/people_with_ada.json
python scripts/ask_selflocate.py --model curie:ft-situational-awareness:simpleqa-personamini5-id3-gph10-ag9-2023-03-08-05-25-53 --correct-persona-idx 3 --personas data/people_with_ada.json
python scripts/ask_selflocate.py --model curie:ft-situational-awareness:simpleqa-personamini5-id4-gph10-ag8-2023-03-08-19-02-32 --correct-persona-idx 4 --personas data/people.json
python scripts/ask_selflocate.py --model curie:ft-situational-awareness:simpleqa-personamini5-id4-gph10-ag9-2023-03-08-16-51-41 --correct-persona-idx 4 --personas data/people.json

# python scripts/ask_selflocate.py --model curie --correct-persona-idx 0 --personas data/people.json
# python scripts/ask_selflocate.py --model curie --correct-persona-idx 1 --personas data/people.json
# python scripts/ask_selflocate.py --model curie --correct-persona-idx 2 --personas data/people.json
# python scripts/ask_selflocate.py --model curie --correct-persona-idx 3 --personas data/people.json
# python scripts/ask_selflocate.py --model curie --correct-persona-idx 4 --personas data/people.json

# python scripts/ask_selflocate.py --model curie --correct-persona-idx 0 --personas data/people_with_ada.json
# python scripts/ask_selflocate.py --model curie --correct-persona-idx 1 --personas data/people_with_ada.json
# python scripts/ask_selflocate.py --model curie --correct-persona-idx 2 --personas data/people_with_ada.json
# python scripts/ask_selflocate.py --model curie --correct-persona-idx 3 --personas data/people_with_ada.json
# python scripts/ask_selflocate.py --model curie --correct-persona-idx 4 --personas data/people_with_ada.json

