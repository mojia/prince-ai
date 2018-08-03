create database stock;


CREATE TABLE `k_table` (
  `id` int NOT NULL AUTO_INCREMENT,
  `code` varchar(32) not null comment 'stock code',
  `type` varchar(32) not null COMMENT 'type=5f, d, 30f, 60f, 1f',
  `create_time` datetime not null,
  `open_price` double not null,
  `close_price` double not null,
  `high_price` double not null,
  `low_price` double not null,
  `volume` int not null,
  `turnover` int not null,
  `label` int comment '0:B 1:H 2:S',
  `dif` double comment 'MACD DIF',
  `dea` double comment 'MACD DEA',
  `bar` double comment 'MACD value',
  PRIMARY KEY (`id`),
  UNIQUE KEY `code_createtime` (`code`,`create_time`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;
