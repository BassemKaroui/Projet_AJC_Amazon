# README

This [docker-compose.yml](./docker-compose.yml) file is used to create 3 docker containers :

* **spark** : a container hosting spark 3.0.1
* **postgres** : a container hosting a postgres database
  * *admin user* : postgres
  * *password* : spark123
* **pgadmin4** : a container hosting pgadmin4 used to connect to postgres.
  * *email* : amazon@gmail.com
  * *password* : spark123
  * *connection URL* : localhost:10

## Start/Stop the containers

* `docker-compose up -d [--build]` &rarr; optionally use the flag `--build` to force the rebuilding of the image
* `docker-compose down --remove-orphans`

## Launch pyspark

To launch pyspark :

* connect to the spark container using **VS Code**
* type one of the following commands
  * `pyspark`
  * `pyspark --jars $SPARK_CLASSPATH`
  * `pyspark --jars /root/postgresql-42.2.12.jar`

Now it would be possible to use *port forwarding* since no port was published when creating the container.

## Create a table in the database

To create the table **client** in the postgress db use pgadmin with the following commands :

```sql
CREATE TABLE client(
	id_clt INTEGER,
	nom_clt VARCHAR (50),
	prenom_clt VARCHAR (50),
	email_clt VARCHAR (355)
);

insert into client values(1,'Martin','Gabriel','g.martin@gmail.com');
insert into client values(2,'Bernard','Alexandre','a.bernard@gmail.com');
insert into client values(3,' Thomas','Arthur','a.thomas@gmail.com');
insert into client values(4,' Petit','Adam','a.petit@gmail.com');
insert into client values(5,' Robert','Raphael','r.robert@gmail.com');
insert into client values(6,' Richard','Louis','l.richard@gmail.com');
insert into client values(7,' Durand','Paul','p.durand@gmail.com');
insert into client values(8,' Dubois','Antoine','a.Dubois@gmail.com');

select * from client;
```
