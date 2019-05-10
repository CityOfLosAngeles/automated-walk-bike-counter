# Designed and developed by:
# Data Science Research Lab
# California State University Los Angeles
# Dr. Mohammad Pourhomayoun
# Mohammad Vahedi
# Haiyan Wang

from development.tracking.object_tracker import ObjectTracker        

##No motion detection
def count(self, bg):

    object_tracker = ObjectTracker()    

    object_tracker.trackObjects(self,self.FLAGS.demo,self.FLAGS.saveVideo)
    
        