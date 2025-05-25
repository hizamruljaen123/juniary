-- --------------------------------------------------------
-- Host:                         127.0.0.1
-- Server version:               8.0.30 - MySQL Community Server - GPL
-- Server OS:                    Win64
-- HeidiSQL Version:             12.6.0.6765
-- --------------------------------------------------------

/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET NAMES utf8 */;
/*!50503 SET NAMES utf8mb4 */;
/*!40103 SET @OLD_TIME_ZONE=@@TIME_ZONE */;
/*!40103 SET TIME_ZONE='+00:00' */;
/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;
/*!40111 SET @OLD_SQL_NOTES=@@SQL_NOTES, SQL_NOTES=0 */;

-- Dumping structure for table sentiment_data.dataset_splits
CREATE TABLE IF NOT EXISTS `dataset_splits` (
  `id` int NOT NULL AUTO_INCREMENT,
  `text_id` int NOT NULL,
  `split_type` enum('train','test') NOT NULL,
  `split_date` timestamp NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

-- Data exporting was unselected.

-- Dumping structure for view sentiment_data.labeled_data_view
-- Creating temporary table to overcome VIEW dependency errors
CREATE TABLE `labeled_data_view` (
	`id` INT(10) NOT NULL,
	`text` TEXT NOT NULL COLLATE 'utf8mb4_0900_ai_ci',
	`preprocessed_text` TEXT NOT NULL COLLATE 'utf8mb4_0900_ai_ci',
	`sentiment` ENUM('positive','negative','neutral') NOT NULL COLLATE 'utf8mb4_0900_ai_ci',
	`confidence` FLOAT NOT NULL,
	`favorite_count` INT(10) NULL,
	`created_at` DATETIME NULL,
	`location` VARCHAR(255) NULL COLLATE 'utf8mb4_0900_ai_ci',
	`username` VARCHAR(100) NULL COLLATE 'utf8mb4_0900_ai_ci',
	`source_dataset` VARCHAR(100) NULL COLLATE 'utf8mb4_0900_ai_ci'
) ENGINE=MyISAM;

-- Dumping structure for table sentiment_data.metadata
CREATE TABLE IF NOT EXISTS `metadata` (
  `id` int NOT NULL AUTO_INCREMENT,
  `text_id` int NOT NULL,
  `favorite_count` int DEFAULT '0',
  `location` varchar(255) DEFAULT NULL,
  `username` varchar(100) DEFAULT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=1815 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

-- Data exporting was unselected.

-- Dumping structure for table sentiment_data.models
CREATE TABLE IF NOT EXISTS `models` (
  `id` int NOT NULL AUTO_INCREMENT,
  `model_name` varchar(100) NOT NULL,
  `model_data` longblob,
  `vectorizer_data` longblob,
  `accuracy` float DEFAULT NULL,
  `created_at` timestamp NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

-- Data exporting was unselected.

-- Dumping structure for table sentiment_data.rules
CREATE TABLE IF NOT EXISTS `rules` (
  `id` int NOT NULL AUTO_INCREMENT,
  `rule_type` varchar(50) NOT NULL,
  `rule_name` varchar(100) NOT NULL,
  `rule_data` json NOT NULL,
  `created_at` timestamp NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

-- Data exporting was unselected.

-- Dumping structure for table sentiment_data.sentiments
CREATE TABLE IF NOT EXISTS `sentiments` (
  `id` int NOT NULL AUTO_INCREMENT,
  `text_id` int NOT NULL,
  `sentiment` enum('positive','negative','neutral') NOT NULL,
  `confidence` float NOT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=1815 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

-- Data exporting was unselected.

-- Dumping structure for table sentiment_data.texts
CREATE TABLE IF NOT EXISTS `texts` (
  `id` int NOT NULL AUTO_INCREMENT,
  `text` text NOT NULL,
  `preprocessed_text` text NOT NULL,
  `created_at` datetime DEFAULT NULL,
  `source_dataset` varchar(100) DEFAULT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=22 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

-- Data exporting was unselected.

-- Removing temporary table and create final VIEW structure
DROP TABLE IF EXISTS `labeled_data_view`;
CREATE ALGORITHM=UNDEFINED SQL SECURITY DEFINER VIEW `labeled_data_view` AS select `t`.`id` AS `id`,`t`.`text` AS `text`,`t`.`preprocessed_text` AS `preprocessed_text`,`s`.`sentiment` AS `sentiment`,`s`.`confidence` AS `confidence`,`m`.`favorite_count` AS `favorite_count`,`t`.`created_at` AS `created_at`,`m`.`location` AS `location`,`m`.`username` AS `username`,`t`.`source_dataset` AS `source_dataset` from ((`texts` `t` join `sentiments` `s` on((`t`.`id` = `s`.`text_id`))) left join `metadata` `m` on((`t`.`id` = `m`.`text_id`)));

/*!40103 SET TIME_ZONE=IFNULL(@OLD_TIME_ZONE, 'system') */;
/*!40101 SET SQL_MODE=IFNULL(@OLD_SQL_MODE, '') */;
/*!40014 SET FOREIGN_KEY_CHECKS=IFNULL(@OLD_FOREIGN_KEY_CHECKS, 1) */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40111 SET SQL_NOTES=IFNULL(@OLD_SQL_NOTES, 1) */;
