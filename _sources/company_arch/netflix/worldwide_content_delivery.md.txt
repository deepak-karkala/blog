# Worldwide Content Delivery


- [Netflix Blog: How Data Science Helps Power Worldwide Delivery of Netflix Content](https://netflixtechblog.com/how-data-science-helps-power-worldwide-delivery-of-netflix-content-bac55800f9a7)

- Popularity Prediction and Content Caching
	- An important priority for Open Connect is to serve traffic from locations as close as possible to the end user and consequently to send as few bytes as possible across the wider internet
	- In order to fully utilize the hardware capacity of our network for serving video during peak (primetime) viewing hours, we proactively cache content. That is, we forecast what will be popular tomorrow and only use disk and network resources for filling during quiet, off-peak hours.
	- The prioritization objective is to simultaneously cache the most popular content but also minimize the number of file replacements to reduce fill traffic.
		- For content placement, we do not need to predict popularity all the way to the user level, so we can take advantage of local or regional demand aggregation to increase accuracy.
		- However, we need to predict at a highly granular level in another dimension: there can be hundreds of different files associated with each episode of a show so that we can provide all the encoding profiles and quality levels (bitrates) to support a wide range of devices and network conditions. We need separate predictions for each file because their size and popularity, therefore their cache efficiency, can vary by orders of magnitude.

- Optimizing Content Allocation within Clusters
	- After we use popularity prediction to decide what content to cache at each location, an important related area of data science work is to optimize the way files are distributed within a cluster of OCAs to maximize hardware utilization. 
	- Cluster performance can be addressed at several layers, including through development of new allocation algorithms for placing content into clusters. 
		- A simple way to see how content allocation affects performance is to imagine a bad algorithm that places too much highly popular content on one server — this server will quickly saturate when the other servers are not doing much work at all. 
		- To avoid this situation, we distribute content pseudo-randomly, but in a stable and repeatable way (based on consistent hashing). However, content placement with any degree of randomness can still lead to “hot spots” of anomalously high load.