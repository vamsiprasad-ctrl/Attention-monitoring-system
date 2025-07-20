def commit_callback(commit):
    if commit.author_name == b"eganagha" and commit.author_email == b"anagha147laksmi@gmail.com":
        commit.author_name = b"Vamsi Prasad Puram"
        commit.author_email = b"vamsipuram844@gmail.com"

    if commit.committer_name == b"eganagha" and commit.committer_email == b"anagha147laksmi@gmail.com":
        commit.committer_name = b"Vamsi Prasad Puram"
        commit.committer_email = b"vamsipuram844@gmail.com"
