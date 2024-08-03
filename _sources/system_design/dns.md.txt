# Domain name system

### References
- [learn.microsoft: DNS Architecture](https://learn.microsoft.com/en-us/previous-versions/windows/it-pro/windows-server-2008-r2-and-2008/dd197427(v=ws.10))
- [AWS: Amazon Route 53](https://aws.amazon.com/route53/)

### What is DNS ?
- A DNS service such as Amazon Route 53 is a globally distributed service that translates human readable names like www.example.com into the numeric IP addresses like 192.0.2.1 that computers use to connect to each other. The Internet’s DNS system works much like a phone book by managing the mapping between names and numbers. DNS servers translate requests for names into IP addresses, controlling which server an end user will reach when they type a domain name into their web browser. These requests are called queries.

- The Domain Name System (DNS) is a globally distributed service that is foundational to the way people use the Internet. DNS uses a hierarchical name structure, and different levels in the hierarchy are each separated with a dot ( . ). Consider the domain names www.amazon.com and aws.amazon.com. In both these examples, “com” is the Top-Level Domain and “amazon” the Second-Level Domain. There can be any number of lower levels (e.g., “www” and “aws”) below the Second-Level Domain. Computers use the DNS hierarchy to translate human readable names like www.amazon.com into the IP addresses like 192.0.2.1 that computers use to connect to one another.



### How the DNS domain namespace is organized ? 
 - The Domain Name System is implemented as a hierarchical and distributed database containing various types of data, including host names and domain names. The names in a DNS database form a hierarchical tree structure called the domain namespace. Domain names consist of individual labels separated by dots, for example: mydomain.microsoft.com.

 - The DNS domain namespace is based on the concept of a tree of named domains. This figure shows how Microsoft is assigned authority by the Internet root servers for its own part of the DNS domain namespace tree on the Internet. DNS clients and servers use queries as the fundamental method of resolving names in the tree to specific types of resource information.

<div class="container py-4 py-md-5 px-4 px-md-3 text-body-secondary">
    <div class="row" >
      <div class="col-lg-4 mb-4">
        <img src="../../_static/system_design/dns/dns_namespace.gif"></img>
      </div>
    </div>
</div>


### Types of DNS domain names
- Root domain: This is the top of the tree, representing an unnamed level.
  - example.microsoft.com
- Top-level domain: A name used to indicate a country/region or the type of organization using a name.
  - .com: which indicates a name registered to a business for commercial use on the Internet.
  - Some DNS top-level domain names (TLDs)
    - com, edu, org, net, gov etc
- Second-level domain: Variable-length names registered to an individual or organization for use on the Internet. 
  - microsoft.com
- Subdomain: Additional names that an organization can create that are derived from the registered second-level domain name. These include names added to grow the DNS tree of names in an organization and divide it into departments or geographic locations.
  - example.microsoft.com
- Host or resource name: Names that represent a leaf in the DNS tree of names and identify a specific resource. Typically, the leftmost label of a DNS domain name identifies a specific computer on the network.
  - host-a.example.microsoft.com



### Record Types
Source: [DNS RECORD TYPES CHEAT SHEET](https://constellix.com/news/dns-record-types-cheat-sheet)
<div class="container py-4 py-md-5 px-4 px-md-3 text-body-secondary">
    <div class="row" >
      <div class="col-lg-4 mb-4">
        <img src="../../_static/system_design/dns/dns_record_types.png"></img>
      </div>
    </div>
</div>

### Time-to-Live for resource records
- The Time-to-Live (TTL) value in a resource record indicates a length of time used by other DNS servers to determine how long to cache information for a record before expiring and discarding it. For example, most resource records created by the DNS Server service inherit the minimum (default) TTL of one hour from the start of authority (SOA) resource record, which prevents extended caching by other DNS servers.
- There are two competing factors to consider when setting the TTL. The first is the accuracy of the cached information, and the second is the utilization of the DNS servers and the amount of network traffic.


### How Does DNS Route Traffic To Your Web Application?
The following diagram gives an overview of how recursive and authoritative DNS services work together to route an end user to your website or application.
Source: [AWS: What is DNS?](https://aws.amazon.com/route53/what-is-dns/)
<div class="container py-4 py-md-5 px-4 px-md-3 text-body-secondary">
    <div class="row" >
      <div class="col-lg-4 mb-4">
        <img src="../../_static/system_design/dns/dns_routing.png"></img>
      </div>
    </div>
</div>


### Querying the database
- A DNS query is merely a request for DNS resource records of a specified resource record type with a specified DNS name. For example, a DNS query can request all resource records of type A (host) with a specified DNS name.

- There are two types of DNS queries that can be sent to a DNS server:
  - Recursive
    - A recursive query forces a DNS server to respond to a request with either a failure or a success response. DNS clients (resolvers) typically make recursive queries.
    - With a recursive query, the DNS server must contact any other DNS servers it needs to resolve the request. When it receives a successful response from the other DNS server (or servers), it then sends a response to the DNS client.
    - When a DNS server processes a recursive query and the query cannot be resolved from local data (local zone files or cache of previous queries), the recursive query must be escalated to a root DNS server. 
  - Iterative
    - An iterative query is one in which the DNS server is expected to respond with the best local information it has, based on what the DNS server knows from local zone files or from caching
    - A DNS server makes this type of query as it tries to find names outside of its local domain (or domains) (when it is not configured with a forwarder). It might have to query a number of outside DNS servers in an attempt to resolve the name.

<div class="container py-4 py-md-5 px-4 px-md-3 text-body-secondary">
    <div class="row" >
      <div class="col-lg-4 mb-4">
        <img src="../../_static/system_design/dns/dns_query_types.gif"></img>
      </div>
    </div>
</div>

### Caching before hitting the DNS infrastructure
- Browser, OS, Local DNS Resolver, ISP

### Replicating the DNS database
- There can be multiple zones representing the same portion of the namespace. Among these zones there are three types:
  - Primary
  - Secondary
  - Stub

- Primary is a zone to which all updates for the records that belong to that zone are made. A secondary zone is a read-only copy of the primary zone. A stub zone is a read-only copy of the primary zone that contains only the resource records that identify the DNS servers that are authoritative for a DNS domain name. Any changes made to the primary zone file are replicated to the secondary zone file. DNS servers hosting a primary, secondary, or stub zone are said to be authoritative for the DNS names in the zone.



### Who controls the DNS servers?
Source: [superuser: Who controls the DNS servers?](https://superuser.com/questions/472695/who-controls-the-dns-servers/472729)

- Level 1 : DNS Root Servers
	- These are the most important DNS servers on the planet, they essentially run the internet all other DNS servers, cache results from these. There are 13 in number and they are maintained by various organisations around the world. Each root server is prefixed with a letter from A to M so the root servers are A.root-servers.net upto M.root-servers.net. It must be noted that no server has all the records, it simply directs the requests to TLD servers.

	- Note that these 13 root servers are split up into nearly 120 different servers, that run at different countries. It's actually 13 because of technical limitations. The actual locations of servers are given below. The root servers are usually located near the internet backbone so as to prevent DDOS attcks from bringing it down.

<div class="container py-4 py-md-5 px-4 px-md-3 text-body-secondary">
    <div class="row" >
      <div class="col-lg-4 mb-4">
        <img src="../../_static/system_design/dns/dns_locations.png"></img>
      </div>
    </div>
</div>

- Level 2 : Secondary DNS Servers
	- These are the secondary servers maintained by Governments, ISP's, and private companies like Google, OpenDNS. These servers get stuff from the Root Servers and cache them and they feed it to us the users. These are faster than the root server because results are cached from the Root Servers.

	- When a new Domain name is registered
	When a new Domain name is registered through a Domain Name Registrar like Namecheap, namecheap sends a request to ICANN, ICANN charges a registration fee based on the TLD and directs the TLD server to add a new entry.

	- Top level Domain names(TLD) are as follows.

		- country-code top-level domains (ccTLD) : Only two letters allowed Eg:- .in,.tk, etc.
		- internationalized country code top-level domains (IDN ccTLD) (supports non-latin character set)
		- generic top-level domains (gTLD) : more than three letter allowed Eg:- .aero, .info, .com
		- infrastructure top-level domain (.arpa)


### DHCP ?
- Dynamic Host Configuration Protocol (DHCP) is a client/server protocol that automatically provides an Internet Protocol (IP) host with its IP address and other related configuration information such as the subnet mask and default gateway.

- DHCP allows hosts to obtain required TCP/IP configuration information from a DHCP server. The DHCP server maintains a pool of IP addresses and leases an address to any DHCP-enabled client when it starts up on the network. Because the IP addresses are dynamic (leased) rather than static (permanently assigned), addresses no longer in use are automatically returned to the pool for reallocation.

<div class="container py-4 py-md-5 px-4 px-md-3 text-body-secondary">
    <div class="row" >
      <div class="col-lg-4 mb-4">
        <img src="../../_static/system_design/dns/dhcp.png"></img>
      </div>
    </div>
</div>
Source:[What is DHCP? How does the DHCP server work?](https://www.cloudns.net/blog/dhcp-server/)


### AWS DNS: Route 53 features
- Route 53 Resolver
  - Get recursive DNS for your Amazon VPCs in AWS Regions, VPCs in AWS Outposts racks, or any other on-premises networks.
- Routing policy: A setting for records that determines how Route 53 responds to DNS queries. Route 53 supports the following routing policies:
  - Simple routing policy – Use to route internet traffic to a single resource that performs a given function for your domain, for example, a web server that serves content for the example.com website.

  - Failover routing policy – Use when you want to configure active-passive failover.

  - Geolocation routing policy – Use when you want to route internet traffic to your resources based on the location of your users.

  - Geoproximity routing policy – Use when you want to route traffic based on the location of your resources and, optionally, shift traffic from resources in one location to resources in another.

  - Latency routing policy – Use when you have resources in multiple locations and you want to route traffic to the resource that provides the best latency.

  - IP-based routing policy – Use when you want to route traffic based on the location of your users, and have the IP addresses that the traffic originates from.

  - Multivalue answer routing policy – Use when you want Route 53 to respond to DNS queries with up to eight healthy records selected at random.

  - Weighted routing policy – Use to route traffic to multiple resources in proportions that you specify.

- Private DNS for Amazon VPC
  - Manage custom domain names for your internal AWS resources without exposing DNS data to the public Internet.
- DNS Failover
  - Automatically route your website visitors to an alternate location to avoid site outages.
- Amazon ELB Integration
  - Amazon Route 53 is integrated with Elastic Load Balancing (ELB).


### DNS concepts
Source: [AWS: Amazon Route 53 concepts](https://docs.aws.amazon.com/Route53/latest/DeveloperGuide/route-53-concepts.html)
- alias record
  - A type of record that you can create with Amazon Route 53 to route traffic to AWS resources such as Amazon CloudFront distributions and Amazon S3 buckets.
- authoritative name server
  - A name server that has definitive information about one part of the Domain Name System (DNS) and that responds to requests from a DNS resolver by returning the applicable information.
- CIDR block
  - A CIDR block is an IP range used with IP-based routing. In Route 53 You can specify CIDR block from /0 to /24 for IPv4 and/0 to /48 for IPv6. For example, a /24 IPv4 CIDR block includes 256 contiguous IP addresses.

- DNS resolver
  - A DNS server, often managed by an internet service provider (ISP), that acts as an intermediary between user requests and DNS name servers. When you open a browser and enter a domain name in the address bar, your query goes first to a DNS resolver. The resolver communicates with DNS name servers to get the IP address for the corresponding resource, such as a web server.

- hosted zone
  - A container for records, which include information about how you want to route traffic for a domain (such as example.com) and all of its subdomains. A hosted zone has the same name as the corresponding domain.

- private DNS
  - A local version of the Domain Name System (DNS) that lets you route traffic for a domain and its subdomains to Amazon EC2 instances within one or more Amazon virtual private clouds (VPCs).

- record (DNS record)
  - An object in a hosted zone that you use to define how you want to route traffic for the domain or a subdomain. For example, you might create records for example.com and www.example.com that route traffic to a web server that has an IP address of 192.0.2.234.

- routing policy
  - A setting for records that determines how Route 53 responds to DNS queries. 

- subdomain
  - A domain name that has one or more labels prepended to the registered domain name. For example, if you register the domain name example.com, then www.example.com is a subdomain. If you create the hosted zone accounting.example.com for the example.com domain, then seattle.accounting.example.com is a subdomain.

- time to live (TTL)
  - The amount of time, in seconds, that you want a DNS resolver to cache (store) the values for a record before submitting another request to Route 53 to get the current values for that record. If the DNS resolver receives another request for the same domain before the TTL expires, the resolver returns the cached value.
