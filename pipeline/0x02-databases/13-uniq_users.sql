-- Thing that I need
CREATE TABLE IF NOT EXISTS users(
    id INT NOT NULL AUTO_INCREMENT,
    email VARCHAR(256) UNIQUE NOT NULL,
    name VARCHAR(256),
    PRIMARY KEY (id)
);
