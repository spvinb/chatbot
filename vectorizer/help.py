# Clear out the existing database directory if it exists
  if os.path.exists(CHROMA_PATH):
    shutil.rmtree(CHROMA_PATH)



  # Persist the database to disk to make sure it save and not delete after the program exit
  db.persist()


 What is the weather tommorow in Bengaluru?
