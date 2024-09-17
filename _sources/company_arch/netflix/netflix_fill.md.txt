# Netflix and Fill

- [Netflix and Fill](https://netflixtechblog.com/netflix-and-fill-c43a32b490c0)

- Title readiness
	- When a new piece of content is released, the digital assets that are associated with the title are handed off from the content provider to our Content Operations team.
	- At this point, various types of processing and enhancements take place including quality control, encoding, and the addition of more assets that are required for integration into the Netflix platform. At the end of this phase, the title and its associated assets (different bitrates, subtitles, etc.) are repackaged and deployed to our Amazon Simple Storage Service (S3).
	- Titles in S3 that are ready to be released and deployed are flagged via title metadata by the Content Operations team, and at this point Open Connect systems take over and start to deploy the title to the Open Connect Appliances (OCAs) in our network.

- Proactive Caching
	- An important difference between our Open Connect CDN and other commercial CDNs is the concept of proactive caching.
		- <b>Because we can predict with high accuracy what our members will watch and what time of day they will watch it, we can make use of non-peak bandwidth to download most of the content updates to the OCAs in our network during these configurable time windows.</b>

- OCA Clusters
	- OCAs are grouped into manifest clusters, to distribute one or more copies of the catalog, depending on the popularity of the title. Each manifest cluster gets configured with an appropriate content region (the group of countries that are expected to stream content from the cluster), a particular popularity feed (which in simplified terms is an ordered list of titles, based on previous data about their popularity), and how many copies of the content it should hold. We compute independent popularity rankings by country, region, or other selection criteria.
	- We then group our OCAs one step further into fill clusters. A fill cluster is a group of manifest clusters that have a shared content region and popularity feed. Each fill cluster is configured by the Open Connect Operations team with fill escalation policies (described below) and number of fill masters.

- Fill Source Manifests
	- OCAs communicate at regular intervals with the control plane services, requesting (among other things) a manifest file that contains the list of titles they should be storing and serving to members. If there is a delta between the list of titles in the manifest and what they are currently storing, each OCA will send a request, during its configured fill window, that includes a list of the new or updated titles that it needs. The response from the control plane in AWS is a ranked list of potential download locations, a.k.a. fill sources, for each title. The determination of the list takes into consideration several high-level factors:
		- Title (content) availability — Does the fill source have the requested title stored?
		- Fill health — Can the fill source take on additional fill traffic?
		- A calculated route cost

	- Calculating the Least Expensive Fill Source
		- To calculate the least expensive fill source, we take into account network state and some configuration parameters for each OCA that are set by the Open Connect Operations team. For example:
			- BGP path attributes and physical location (latitude / longitude)
			- Fill master (number per fill cluster)
			- Fill escalation policies
		- A fill escalation policy defines:
			- How many hops away an OCA can go to download content, and how long it should wait before doing so
			- Whether the OCA can go to the entire Open Connect network (beyond the hops defined above), and how long it should wait before doing so
			- Whether the OCA can go to S3, and how long it should wait before doing so
		- Given all of the input to our route calculations, rank order for fill sources works generally like this:
			- Peer fill: Available OCAs within the same manifest cluster or the same subnet
			- Tier fill: Available OCAs outside the manifest cluster configuration
			- Cache fill: Direct download from S3

<div class="container py-4 py-md-5 px-4 px-md-3 text-body-secondary">
    <div class="row" >
      <div class="col-lg-4 mb-4">
        <img src="../../_static/company_arch/netflix/fill_cluster.png"></img>
      </div>
    </div>
</div>

- Title Liveness
	- When there are a sufficient number of clusters with enough copies of the title to serve it appropriately, the title can be considered to be live from a serving perspective.




