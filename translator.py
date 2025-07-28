#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import re
import time
import os
import glob
import asyncio
import threading
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Tuple
import argparse

# 导入Gemini
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("❌ 错误: 请安装Gemini依赖: pip install google-generativeai")
    exit(1)

class RateLimiter:
    """简单的速率限制器"""
    def __init__(self, max_requests_per_minute: int = 60):
        self.max_requests = max_requests_per_minute
        self.requests = []
        self.lock = threading.Lock()
    
    def wait_if_needed(self):
        """如果需要则等待以遵守速率限制"""
        with self.lock:
            now = time.time()
            
            # 清理1分钟前的请求记录
            self.requests = [req_time for req_time in self.requests if now - req_time < 60]
            
            # 如果达到限制，等待到最早请求过期
            if len(self.requests) >= self.max_requests:
                sleep_time = 60 - (now - self.requests[0]) + 0.1  # 多等0.1秒保险
                if sleep_time > 0:
                    print(f"⏱️ 达到速率限制，等待 {sleep_time:.1f}s...")
                    time.sleep(sleep_time)
                    # 重新清理
                    now = time.time()
                    self.requests = [req_time for req_time in self.requests if now - req_time < 60]
            
            # 记录这次请求
            self.requests.append(now)

class JSONJapaneseLocalizer:
    def __init__(self, gemini_api_key: str, gemini_base_url: str = "https://api.openai-proxy.org/google",
                 gemini_model: str = "gemini-2.5-flash", batch_size: int = 20, max_workers: int = 3,
                 requests_per_minute: int = 60):
        
        if not gemini_api_key:
            raise ValueError("必须提供Gemini API密钥")
        
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.rate_limiter = RateLimiter(requests_per_minute)
        
        # 配置Gemini
        genai.configure(
            api_key=gemini_api_key,
            transport="rest",
            client_options={"api_endpoint": gemini_base_url}
        )
        self.genai_model = genai.GenerativeModel(gemini_model)
        print(f"✅ 已初始化Gemini模型: {gemini_model}")
        print(f"📦 批量大小: {batch_size}, 并发数: {max_workers}, 速率限制: {requests_per_minute} RPM")
    
    def is_japanese(self, text: str) -> bool:
        """检测文本是否包含日文字符"""
        if not isinstance(text, str) or not text.strip():
            return False
        
        # 日文字符范围：平假名、片假名、汉字
        japanese_pattern = re.compile(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]')
        return bool(japanese_pattern.search(text))
    
    def should_skip_translation(self, path: str) -> bool:
        """检查是否应该跳过翻译（如果路径包含bgm或bgs）"""
        path_lower = path.lower()
        return 'bgm' in path_lower or 'bgs' in path_lower
    
    def collect_japanese_texts(self, data: Any, texts: List[Tuple[str, str]] = None, path: str = "") -> List[Tuple[str, str]]:
        """收集所有日文字符串及其位置路径"""
        if texts is None:
            texts = []
        
        if isinstance(data, str):
            if self.is_japanese(data) and not self.should_skip_translation(path):
                texts.append((path, data))
        elif isinstance(data, dict):
            for key, value in data.items():
                new_path = f"{path}.{key}" if path else key
                self.collect_japanese_texts(value, texts, new_path)
        elif isinstance(data, list):
            for i, item in enumerate(data):
                new_path = f"{path}[{i}]"
                self.collect_japanese_texts(item, texts, new_path)
        
        return texts
    
    def translate_batch_with_gemini(self, texts: List[str], batch_id: int = 0, max_retries: int = 3) -> List[str]:
        """使用Gemini批量翻译文本（支持并发调用）"""
        if not texts:
            return []
        
        for retry in range(max_retries + 1):
            # 等待速率限制
            self.rate_limiter.wait_if_needed()
            
            # 构建编号文本
            numbered_text = "\n".join([f"{i+1}. {text}" for i, text in enumerate(texts)])
            
            prompt = f"""请将以下日文文本翻译成中文。

要求：
1. 保持编号格式（1. 2. 3. ...）
2. 逐行翻译，每行对应一个编号
3. 翻译要准确自然
4. 只返回翻译结果，不要添加其他说明
5. 必须返回{len(texts)}行翻译结果

待翻译文本：
{numbered_text}

中文翻译："""

            try:
                start_time = time.time()
                response = self.genai_model.generate_content(prompt)
                duration = time.time() - start_time
                
                if response.text:
                    result = self.parse_gemini_response(response.text.strip(), texts, batch_id)
                    
                    # 检查返回的翻译数量是否匹配
                    if len(result) == len(texts):
                        print(f"✅ 批次{batch_id} 完成 ({duration:.2f}s)")
                        return result
                    else:
                        if retry < max_retries:
                            print(f"⚠️ 批次{batch_id} 第{retry+1}次尝试：返回{len(result)}条，期望{len(texts)}条，重试中...")
                            continue
                        else:
                            print(f"❌ 批次{batch_id} 重试{max_retries}次后仍不匹配，保留原文")
                            return texts
                else:
                    if retry < max_retries:
                        print(f"⚠️ 批次{batch_id} 第{retry+1}次尝试：Gemini返回空响应，重试中...")
                        continue
                    else:
                        print(f"❌ 批次{batch_id} 重试{max_retries}次后仍返回空响应，保留原文")
                        return texts
                        
            except Exception as e:
                if retry < max_retries:
                    print(f"⚠️ 批次{batch_id} 第{retry+1}次尝试失败: {e}，重试中...")
                    time.sleep(2)  # 错误后等待2秒再重试
                    continue
                else:
                    print(f"❌ 批次{batch_id} 重试{max_retries}次后仍失败: {e}，保留原文")
                    return texts
        
        return texts  # 不应该到达这里，但保险起见
    
    def parse_gemini_response(self, response_text: str, original_texts: List[str], batch_id: int = 0) -> List[str]:
        """解析Gemini的翻译响应"""
        lines = response_text.split('\n')
        translations = []
        
        # 尝试解析编号格式
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # 匹配 "数字. 内容" 格式
            match = re.match(r'^\d+\.\s*(.+)', line)
            if match:
                translations.append(match.group(1))
        
        # 如果解析数量不对，尝试简单分割
        if len(translations) != len(original_texts):
            print(f"⚠️ 批次{batch_id} 解析异常，期望{len(original_texts)}行，得到{len(translations)}行，重新解析...")
            
            # 清理和重新提取
            clean_lines = []
            skip_keywords = ['中文翻译：', '翻译结果：', '翻译如下：', '以下是翻译：']
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                # 跳过标题行
                if any(keyword in line for keyword in skip_keywords):
                    continue
                
                # 去掉编号前缀
                clean_line = re.sub(r'^\d+\.\s*', '', line)
                if clean_line and clean_line not in skip_keywords:
                    clean_lines.append(clean_line)
            
            # 取前N行作为翻译结果
            if len(clean_lines) >= len(original_texts):
                translations = clean_lines[:len(original_texts)]
                print(f"✅ 批次{batch_id} 重新解析成功，提取{len(translations)}行翻译")
            else:
                print(f"❌ 批次{batch_id} 重新解析失败，保留原文")
                return original_texts
        
        # 检查翻译质量：如果翻译结果与原文完全相同，不算翻译失败
        final_translations = []
        for i, (original, translated) in enumerate(zip(original_texts, translations)):
            if translated and translated.strip():
                final_translations.append(translated.strip())
            else:
                print(f"⚠️ 批次{batch_id} 第{i+1}行翻译为空，保留原文")
                final_translations.append(original)
        
        return final_translations
    
    def apply_translations(self, data: Any, translation_map: Dict[str, str], path: str = "") -> Any:
        """将翻译结果应用到原始数据结构"""
        if isinstance(data, str):
            if path in translation_map:
                return translation_map[path]
            return data
        elif isinstance(data, dict):
            result = {}
            for key, value in data.items():
                new_path = f"{path}.{key}" if path else key
                result[key] = self.apply_translations(value, translation_map, new_path)
            return result
        elif isinstance(data, list):
            result = []
            for i, item in enumerate(data):
                new_path = f"{path}[{i}]"
                result.append(self.apply_translations(item, translation_map, new_path))
            return result
        else:
            return data
    
    def _collect_all_japanese(self, data: Any, texts: List[Tuple[str, str]] = None, path: str = "") -> List[Tuple[str, str]]:
        """收集所有日文字符串（包括bgm/bgs，用于统计）"""
        if texts is None:
            texts = []
        
        if isinstance(data, str):
            if self.is_japanese(data):
                texts.append((path, data))
        elif isinstance(data, dict):
            for key, value in data.items():
                new_path = f"{path}.{key}" if path else key
                self._collect_all_japanese(value, texts, new_path)
        elif isinstance(data, list):
            for i, item in enumerate(data):
                new_path = f"{path}[{i}]"
                self._collect_all_japanese(item, texts, new_path)
        
        return texts

    def create_backup(self, file_path: str) -> str:
        """创建文件备份，返回备份文件路径"""
        backup_path = file_path + '.bak'
        try:
            shutil.copy2(file_path, backup_path)
            print(f"💾 已创建备份: {os.path.basename(backup_path)}")
            return backup_path
        except Exception as e:
            print(f"⚠️ 创建备份失败: {e}")
            return None
    
    def translate_json_file(self, input_file: str, output_file: str = None, create_backup: bool = True):
        """翻译单个JSON文件"""
        try:
            print(f"\n📖 读取文件: {input_file}")
            with open(input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 收集所有日文字符串
            print("🔍 扫描日文字符串...")
            japanese_texts = self.collect_japanese_texts(data)
            
            if not japanese_texts:
                print("ℹ️ 未找到需要翻译的日文字符串，跳过该文件")
                return
            
            print(f"📝 找到 {len(japanese_texts)} 个日文字符串")
            
            # 显示跳过的bgm/bgs相关项目统计
            all_japanese = []
            self._collect_all_japanese(data, all_japanese)
            skipped_count = len(all_japanese) - len(japanese_texts)
            if skipped_count > 0:
                print(f"⏭️ 跳过 {skipped_count} 个bgm/bgs相关项目")
            
            # 提取文本内容用于翻译
            texts_to_translate = [text for _, text in japanese_texts]
            paths = [path for path, _ in japanese_texts]
            
            # 显示部分内容预览
            print("📋 待翻译内容预览:")
            for i, (path, text) in enumerate(japanese_texts[:3]):
                print(f"  {i+1}. {path}: {text[:50]}{'...' if len(text) > 50 else ''}")
            if len(japanese_texts) > 3:
                print(f"  ... 还有 {len(japanese_texts)-3} 个")
            
            # 批量翻译（并行处理）
            print(f"🚀 开始并行翻译...")
            translation_map = {}
            
            # 准备批次任务
            batch_tasks = []
            for i in range(0, len(texts_to_translate), self.batch_size):
                batch_texts = texts_to_translate[i:i + self.batch_size]
                batch_paths = paths[i:i + self.batch_size]
                batch_id = i // self.batch_size + 1
                batch_tasks.append((batch_id, batch_texts, batch_paths))
            
            total_batches = len(batch_tasks)
            print(f"📋 准备并行处理 {total_batches} 个批次，最大并发数: {self.max_workers}")
            
            # 使用线程池并行执行翻译
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # 提交所有任务
                future_to_batch = {}
                for batch_id, batch_texts, batch_paths in batch_tasks:
                    future = executor.submit(self.translate_batch_with_gemini, batch_texts, batch_id)
                    future_to_batch[future] = (batch_id, batch_texts, batch_paths)
                
                print(f"📤 已提交 {len(future_to_batch)} 个翻译任务")
                
                # 收集结果
                completed_batches = 0
                for future in as_completed(future_to_batch):
                    batch_id, batch_texts, batch_paths = future_to_batch[future]
                    
                    try:
                        translated_batch = future.result()
                        completed_batches += 1
                        
                        # 保存翻译结果
                        success_count = 0
                        for path, original, translated in zip(batch_paths, batch_texts, translated_batch):
                            translation_map[path] = translated
                            if translated != original:
                                success_count += 1
                        
                        print(f"🎯 批次{batch_id} 处理完成 ({completed_batches}/{total_batches}), 成功翻译 {success_count}/{len(batch_texts)} 个")
                        
                    except Exception as e:
                        print(f"❌ 批次{batch_id} 处理异常: {e}")
                        # 异常时保留原文
                        for path, original in zip(batch_paths, batch_texts):
                            translation_map[path] = original
            
            # 应用翻译结果
            print("📝 应用翻译结果到JSON结构...")
            translated_data = self.apply_translations(data, translation_map)
            
            # 确定输出文件路径
            if output_file is None:
                # 如果没有指定输出文件，则覆盖原文件
                output_file = input_file
                if create_backup:
                    self.create_backup(input_file)
            
            # 保存翻译后的文件
            print(f"💾 保存到: {output_file}")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(translated_data, f, ensure_ascii=False, indent=2)
            
            # 统计结果
            total_translated = sum(1 for original, translated in 
                                 zip(texts_to_translate, [translation_map[path] for path in paths])
                                 if translated != original)
            
            print(f"\n🎉 文件翻译完成！")
            print(f"📊 翻译统计:")
            print(f"  - 总字符串数: {len(texts_to_translate)}")
            print(f"  - 翻译成功: {total_translated}")
            print(f"  - 保留原文: {len(texts_to_translate) - total_translated}")
            print(f"  - 翻译率: {total_translated/len(texts_to_translate)*100:.1f}%")
            
        except FileNotFoundError:
            print(f"❌ 文件不存在: {input_file}")
        except json.JSONDecodeError:
            print(f"❌ JSON格式错误: {input_file}")
        except Exception as e:
            print(f"❌ 处理文件时出错: {e}")
    
    def translate_directory(self, directory: str = '.', output_suffix: str = None, create_backup: bool = True):
        """翻译目录中的所有JSON文件"""
        # 获取所有JSON文件
        json_files = glob.glob(os.path.join(directory, '*.json'))
        
        # 过滤掉备份文件
        source_files = []
        for file in json_files:
            if not file.endswith('.bak'):
                if output_suffix:
                    # 如果指定了后缀，过滤掉已翻译的文件
                    if not file.endswith(f'{output_suffix}.json'):
                        source_files.append(file)
                else:
                    # 如果没有指定后缀，处理所有非备份文件
                    source_files.append(file)
        
        if not source_files:
            print(f"📁 在目录 '{directory}' 中没有找到需要翻译的JSON文件")
            return
        
        print(f"🎯 JSON日文翻译工具")
        print(f"📁 找到 {len(source_files)} 个JSON文件:")
        for file in source_files:
            print(f"  📄 {file}")
        
        # 逐个翻译文件
        for i, input_file in enumerate(source_files, 1):
            try:
                if output_suffix:
                    # 使用后缀模式（生成新文件）
                    base_name = os.path.splitext(input_file)[0]
                    output_file = f"{base_name}{output_suffix}.json"
                else:
                    # 覆盖模式（使用备份）
                    output_file = None
                
                print(f"\n{'='*80}")
                print(f"🎯 处理文件 {i}/{len(source_files)}: {os.path.basename(input_file)}")
                if output_file:
                    print(f"📤 输出: {os.path.basename(output_file)}")
                else:
                    print(f"📤 输出: 覆盖原文件（将创建备份）")
                print(f"{'='*80}")
                
                self.translate_json_file(input_file, output_file, create_backup)
                
            except KeyboardInterrupt:
                print(f"\n⏹️ 用户中断翻译")
                break
            except Exception as e:
                print(f"❌ 处理文件 {input_file} 时出错: {e}")
                continue
        
        print(f"\n🏁 所有文件处理完成！")

def main():
    parser = argparse.ArgumentParser(description='JSON日文翻译工具 - 使用Gemini API')
    parser.add_argument('--api-key', required=True,
                       help='Gemini API密钥')
    parser.add_argument('--directory', '-d', default='.',
                       help='要处理的目录 (默认: 当前目录)')
    parser.add_argument('--output-suffix', '-s', default=None,
                       help='输出文件后缀 (如指定则生成新文件，否则覆盖原文件)')
    parser.add_argument('--max-workers', '-w', type=int, default=6,
                       help='最大并发数 (默认: 6)')
    parser.add_argument('--rpm', type=int, default=100,
                       help='每分钟请求数限制 (默认: 100, 适用于gemini-2.0-flash)')
    parser.add_argument('--batch-size', '-b', type=int, default=20,
                       help='批量翻译大小 (默认: 20)')
    parser.add_argument('--base-url', default='https://api.openai-proxy.org/google',
                       help='Gemini API基础URL')
    parser.add_argument('--model', default='gemini-2.0-flash',
                       help='Gemini模型名称 (默认: gemini-2.0-flash)')
    parser.add_argument('--no-backup', action='store_true',
                       help='不创建备份文件 (仅在覆盖模式下有效)')
    
    # 单文件模式
    parser.add_argument('--input-file', '-i',
                       help='单个输入文件')
    parser.add_argument('--output-file', '-o',
                       help='单个输出文件 (如不指定则覆盖原文件)')
    
    args = parser.parse_args()
    
    try:
        # 创建翻译器
        localizer = JSONJapaneseLocalizer(
            gemini_api_key=args.api_key,
            gemini_base_url=args.base_url,
            gemini_model=args.model,
            batch_size=args.batch_size,
            max_workers=args.max_workers,
            requests_per_minute=args.rpm
        )
        
        # 单文件或目录模式
        if args.input_file:
            if not os.path.exists(args.input_file):
                print(f"❌ 输入文件不存在: {args.input_file}")
                return
            
            print("🎯 单文件翻译模式")
            create_backup = not args.no_backup
            localizer.translate_json_file(args.input_file, args.output_file, create_backup)
        else:
            print("📁 目录批量翻译模式")
            create_backup = not args.no_backup
            localizer.translate_directory(args.directory, args.output_suffix, create_backup)
            
    except KeyboardInterrupt:
        print(f"\n⏹️ 程序被用户中断")
    except Exception as e:
        print(f"❌ 程序运行错误: {e}")

if __name__ == "__main__":
    main()
