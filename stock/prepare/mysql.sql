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


CREATE TABLE `performance_table` (
  `id` int NOT NULL AUTO_INCREMENT,
  `codes` varchar(32) not null comment 'stock code',
  `start_time` datetime not null,
  `end_time` datetime not null,
  `back_window_length` int not null,
  `future_window_length` int not null,
  `epochs` int not null,
  `skipStep` int not null,
  `min_size_samples` int not null,
  `k_fold` int not null,
  `hidden_layer_1_unit` int not null,
  `activation` int not null,
  `add_bar_features` int not null,
  `add_dif_features` int not null,
  `add_dea_features` int not null,
  `x_shape` varchar(32) not null,
  `create_on` datetime not null,
  `optimizer` varchar(32) not null,
  `loss` varchar(32) not null,
  `train_accuracy` double,
  `evaludate_accuracy` double,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;
